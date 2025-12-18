import cv2
import numpy as np
import time
import os
import urllib.request
import threading
from utils.frameGrabber import FrameGrabber
from utils.boundingBox import *

TARGET_W = 1600
TARGET_H = 900

# Red neuronal
DNN_PROTO = "MobileNetSSD_deploy.prototxt"
DNN_MODEL = "MobileNetSSD_deploy.caffemodel"
DNN_CONFIDENCE = 0.5
DNN_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

def download_model_if_needed():
    if not os.path.exists(DNN_PROTO):
        print(f"Descargando {DNN_PROTO}...")
        try:
            url_proto = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"
            urllib.request.urlretrieve(url_proto, DNN_PROTO)
        except Exception as e:
            print(f"Error descargando prototxt: {e}")
    
    if not os.path.exists(DNN_MODEL):
        print(f"Descargando {DNN_MODEL}...")
        try:
            url_model = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel" 
            urllib.request.urlretrieve(url_model, DNN_MODEL)
        except Exception as e:
            print(f"Error descargando caffemodel: {e}")

def merge_rectangles(rects, threshold=40):
    if not rects:
        return []
    
    merged = [list(r) for r in rects]
    
    while True:
        new_merged = []
        used = [False] * len(merged)
        changed = False
        
        for i in range(len(merged)):
            if used[i]:
                continue
            
            current = merged[i]
            used[i] = True
            
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                
                other = merged[j]
                
                if (current[0] - threshold < other[2] and
                    current[2] + threshold > other[0] and
                    current[1] - threshold < other[3] and
                    current[3] + threshold > other[1]):
                    
                    current[0] = min(current[0], other[0])
                    current[1] = min(current[1], other[1])
                    current[2] = max(current[2], other[2])
                    current[3] = max(current[3], other[3])
                    
                    used[j] = True
                    changed = True
            
            new_merged.append(current)
        
        merged = new_merged
        if not changed:
            break
    
    return [tuple(r) for r in merged]

def merge_bboxes_enclosing(bboxes, pad=0, iou_thr=0.15, near_thr=50):
    if not bboxes:
        return []

    # normalizar a listas mutables
    boxes = [list(b) for b in bboxes]

    def is_close_or_overlap(a, b):
        # “cerca” estilo merge_rectangles
        close = (a[0] - near_thr < b[2] and a[2] + near_thr > b[0] and
                 a[1] - near_thr < b[3] and a[3] + near_thr > b[1])
        if close:
            return True
        # overlap por IoU
        return iou(tuple(a), tuple(b)) >= iou_thr

    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        # BFS/DFS para armar el componente conexo
        stack = [i]
        used[i] = True
        group = [boxes[i]]

        while stack:
            k = stack.pop()
            for j in range(len(boxes)):
                if used[j]:
                    continue
                if is_close_or_overlap(boxes[k], boxes[j]):
                    used[j] = True
                    stack.append(j)
                    group.append(boxes[j])

        # bbox envolvente del grupo
        x0 = min(g[0] for g in group) - pad
        y0 = min(g[1] for g in group) - pad
        x1 = max(g[2] for g in group) + pad
        y1 = max(g[3] for g in group) + pad
        merged.append((int(x0), int(y0), int(x1), int(y1)))

    return merged


def check_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea > 0

def bbox_union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def appeared_new_region(curr_bboxes, prev_bboxes, iou_thr=0.20, min_area=800):
    curr = [b for b in curr_bboxes if (b[2]-b[0]) * (b[3]-b[1]) >= min_area]
    prev = [b for b in prev_bboxes if (b[2]-b[0]) * (b[3]-b[1]) >= min_area]

    if not prev and curr:
        return True
    if not curr:
        return False

    for cb in curr:
        best = 0.0
        for pb in prev:
            best = max(best, iou(cb, pb))
        if best < iou_thr:
            return True
    return False

def bbox_center(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

def clamp_bbox(b, w, h):
    x0, y0, x1, y1 = b
    x0 = max(0, min(w-1, x0))
    y0 = max(0, min(h-1, y0))
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    if x1 <= x0: x1 = min(w-1, x0+1)
    if y1 <= y0: y1 = min(h-1, y0+1)
    return (int(x0), int(y0), int(x1), int(y1))

def point_in_bbox(p, b, margin=0):
    x, y = p
    return (b[0]-margin) <= x <= (b[2]+margin) and (b[1]-margin) <= y <= (b[3]+margin)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

feature_params = dict(
    maxCorners=300,
    qualityLevel=0.15,
    minDistance=10,
    blockSize=7
)

MOTION_THRESH = 15
MIN_FEATURES = 10
DILATE_ITER = 4
MAX_DISPLACEMENT = 1000.0
MIN_MOVEMENT_HISTORY = 5.0
STATIC_EDGE_THRESHOLD = 3
CANNY_SENS = 0.33
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
BLUR_KERNEL = 5
DISPLAY_SCALE = 0.5
MIN_CONTOUR_AREA = 100
STATIC_CONTOUR_FRAMES = 8
STATIC_MATCH_DIST = 10.0
STATIC_SIZE_TOL = 0.4
STATIC_FORGET_FRAMES = 60


calibration_mode = False
show_debug_info = False
manual_sens = int(CANNY_SENS * 100)
manual_clip = int(CLAHE_CLIP_LIMIT * 10)
manual_tile = CLAHE_TILE_SIZE
manual_blur = BLUR_KERNEL


def get_dynamic_bounding_box(dynamic_contours):
    if len(dynamic_contours) == 0:
        return None
    
    x_min = float('inf')
    y_min = float('inf')
    x_max = 0
    y_max = 0
    
    for cnt in dynamic_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def get_dynamic_bounding_boxes(dynamic_contours):
    bboxes = []
    for cnt in dynamic_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((int(x), int(y), int(x + w), int(y + h)))
    return bboxes

def filter_points_by_bbox(points, bbox, margin=10):
    if points is None or len(points) == 0:
        return np.array([], dtype=bool)
    
    if bbox is None:
        return np.zeros(len(points), dtype=bool)
    
    x_min, y_min, x_max, y_max = bbox
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin
    
    points_flat = points.reshape(-1, 2)
    valid = (
        (points_flat[:, 0] >= x_min) &
        (points_flat[:, 0] <= x_max) &
        (points_flat[:, 1] >= y_min) &
        (points_flat[:, 1] <= y_max)
    )
    return valid

def filter_points_by_bboxes(points, bboxes, margin=10):
    if points is None or len(points) == 0:
        return np.array([], dtype=bool)
    
    if not bboxes:
        return np.zeros(len(points), dtype=bool)
    
    points_flat = points.reshape(-1, 2)
    valid_total = np.zeros(len(points_flat), dtype=bool)
    
    for (x_min, y_min, x_max, y_max) in bboxes:
        # Expandir bbox con margen
        x0 = x_min - margin
        y0 = y_min - margin
        x1 = x_max + margin
        y1 = y_max + margin
        
        inside = (
            (points_flat[:, 0] >= x0) &
            (points_flat[:, 0] <= x1) &
            (points_flat[:, 1] >= y0) &
            (points_flat[:, 1] <= y1)
        )
        valid_total |= inside  # OR: punto válido si cae en alguna bbox
    
    return valid_total


def nothing(x):
    """Callback vacío para los trackbars"""
    pass

def create_calibration_window():
    cv2.namedWindow('Calibración Canny')
    cv2.createTrackbar('Sensibilidad x100', 'Calibración Canny', manual_sens, 100, nothing)
    cv2.createTrackbar('CLAHE Clip x10', 'Calibración Canny', manual_clip, 100, nothing)
    cv2.createTrackbar('CLAHE Tile', 'Calibración Canny', manual_tile, 16, nothing)
    cv2.createTrackbar('Blur Kernel', 'Calibración Canny', manual_blur, 31, nothing)
    
    info = np.zeros((200, 500, 3), dtype=np.uint8)
    cv2.putText(info, 'Ajusta los parametros:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info, 'Sensibilidad: 0-100 (x0.01)', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(info, 'CLAHE Clip: 0-100 (x0.1)', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(info, 'CLAHE Tile: 2-16', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(info, 'Blur Kernel: 1-31 (impar)', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow('Calibración Canny', info)

def draw_debug_info(frame, brightness, sens, blur, clip, tile, num_points_motion, num_points_canny, num_contours):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
    frame_with_info = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    y_offset = 35
    line_height = 30
    
    cv2.putText(frame_with_info, "DEBUG INFO (tecla 'd' para ocultar)", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(frame_with_info, f"Brillo: {brightness:.1f}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame_with_info, f"Sensibilidad: {sens:.3f}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame_with_info, f"Blur Kernel: {blur}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame_with_info, f"CLAHE Clip: {clip:.1f}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame_with_info, f"CLAHE Tile: {tile}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame_with_info, f"Puntos Motion: {num_points_motion}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y_offset += line_height
    
    cv2.putText(frame_with_info, f"Puntos Canny: {num_points_canny}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame_with_info, f"Contornos dinamicos: {num_contours}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame_with_info


def get_calibration_params():
    global manual_sens, manual_clip, manual_tile, manual_blur
    
    manual_sens = cv2.getTrackbarPos('Sensibilidad x100', 'Calibración Canny')
    manual_clip = cv2.getTrackbarPos('CLAHE Clip x10', 'Calibración Canny')
    manual_tile = max(2, cv2.getTrackbarPos('CLAHE Tile', 'Calibración Canny'))
    manual_blur = cv2.getTrackbarPos('Blur Kernel', 'Calibración Canny')
    
    if manual_blur % 2 == 0:
        manual_blur += 1
    manual_blur = max(1, manual_blur)
    
    sens = manual_sens / 100.0
    clip = manual_clip / 10.0
    tile = manual_tile
    blur = manual_blur
    
    return sens, blur, clip, tile


def calculate_brightness(frame):
    return float(np.mean(frame))

def adjust_canny_params(brightness):
    brightness = min(brightness, 170)
    t = brightness / 170.0
    
    sens = 0.33 + t * 0.17
    
    blur_float = 19.0 + t * 4.0
    blur = int(round(blur_float))
    if blur % 2 == 0:
        blur += 1
    blur = max(1, min(31, blur))
    
    clip = 4.0 - t * 1.0
    
    tile_float = 8.0 - t * 4.0
    tile = int(round(tile_float))
    tile = max(2, min(16, tile))
    
    return sens, blur, clip, tile


def canny_mejorado(img, sens=0.33, clip_limit=2.0, tile_size=8, blur_kernel=5):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    equalized = clahe.apply(img)
    blurred = cv2.GaussianBlur(equalized, (blur_kernel, blur_kernel), 0)
    
    med = np.median(blurred)
    low_thresh = int(max(0, (1.0 - sens) * med))
    high_thresh = int(min(255, (1.0 + sens) * med))
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    return edges

def update_static_contours(closed_contours, static_contours, frame_idx):
    dynamic_contours = []

    for cnt in closed_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0

        best_idx = None
        best_dist = 1e9

        for i, sc in enumerate(static_contours):
            dist = np.hypot(cx - sc["cx"], cy - sc["cy"])
            if dist > STATIC_MATCH_DIST:
                continue

            if sc["w"] == 0 or sc["h"] == 0:
                continue

            if abs(w - sc["w"]) > sc["w"] * STATIC_SIZE_TOL:
                continue
            if abs(h - sc["h"]) > sc["h"] * STATIC_SIZE_TOL:
                continue

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            sc = static_contours[best_idx]
            sc["cx"] = 0.5 * (sc["cx"] + cx)
            sc["cy"] = 0.5 * (sc["cy"] + cy)
            sc["w"] = 0.5 * (sc["w"] + w)
            sc["h"] = 0.5 * (sc["h"] + h)
            sc["life"] += 1
            sc["last_seen"] = frame_idx

            is_static = sc["life"] >= STATIC_CONTOUR_FRAMES
        else:
            static_contours.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "w": float(w),
                    "h": float(h),
                    "life": 1,
                    "last_seen": frame_idx,
                }
            )
            is_static = False

        if not is_static:
            dynamic_contours.append(cnt)

    static_contours[:] = [
        sc
        for sc in static_contours
        if frame_idx - sc["last_seen"] <= STATIC_FORGET_FRAMES
    ]

    return dynamic_contours



def select_input_source():
    print("\n=== SELECCIÓN DE FUENTE ===")
    print("1. Cámara (webcam)")
    print("2. Video desde carpeta Video/")
    
    while True:
        choice = input("Seleccione una opción (1 o 2): ").strip()
        if choice == "1":
            return 0
        elif choice == "3":
            return "rtsp://admin:12345678Ab@192.168.1.11:554/1/1?transport=tcp"
        elif choice == "2":
            video_folder = "Video"
            if not os.path.exists(video_folder):
                print(f"ERROR: La carpeta '{video_folder}' no existe.")
                continue
            
            videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if not videos:
                print(f"No se encontraron videos en la carpeta '{video_folder}'")
                continue
            
            print("\nVideos disponibles:")
            for i, vid in enumerate(videos, 1):
                print(f"{i}. {vid}")
            
            vid_choice = input(f"Seleccione un video (1-{len(videos)}): ").strip()
            try:
                vid_idx = int(vid_choice) - 1
                if 0 <= vid_idx < len(videos):
                    return os.path.join(video_folder, videos[vid_idx])
                else:
                    print("Opción inválida")
            except ValueError:
                print("Entrada inválida")
        else:
            print("Opción inválida. Por favor, seleccione 1 o 2.")


def ensure_points_in_bboxes(prev_gray, mask_img, p0, movement_hist, bboxes,
                            min_pts_per_bbox=6, max_new_per_bbox=25, margin=10):
    if not bboxes:
        return p0, movement_hist

    if p0 is None or len(p0) == 0:
        points_flat = np.empty((0, 2), dtype=np.float32)
    else:
        points_flat = p0.reshape(-1, 2)

    fp = dict(feature_params)
    fp.pop("maxCorners", None)

    for (x0, y0, x1, y1) in bboxes:
        if points_flat.shape[0] > 0:
            inside = (
                (points_flat[:, 0] >= x0) & (points_flat[:, 0] <= x1) &
                (points_flat[:, 1] >= y0) & (points_flat[:, 1] <= y1)
            )
            count_inside = int(np.count_nonzero(inside))
        else:
            count_inside = 0

        if count_inside >= min_pts_per_bbox:
            continue

        roi = np.zeros_like(mask_img)
        xx0 = max(0, x0 - margin); yy0 = max(0, y0 - margin)
        xx1 = min(mask_img.shape[1]-1, x1 + margin); yy1 = min(mask_img.shape[0]-1, y1 + margin)
        cv2.rectangle(roi, (xx0, yy0), (xx1, yy1), 255, -1)
        roi = cv2.bitwise_and(roi, mask_img)

        new_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=roi,
            maxCorners=max_new_per_bbox,
            **fp
        )

        if new_pts is None or len(new_pts) == 0:
            continue

        if p0 is None or len(p0) == 0:
            p0 = new_pts
            movement_hist = np.zeros(len(p0), dtype=np.float32)
        else:
            p0 = np.vstack([p0, new_pts])
            if movement_hist is None:
                movement_hist = np.zeros(len(p0), dtype=np.float32)
            else:
                movement_hist = np.concatenate([movement_hist, np.zeros(len(new_pts), dtype=np.float32)])

        points_flat = p0.reshape(-1, 2)

    return p0, movement_hist


def main():
    global calibration_mode, show_debug_info
    
    download_model_if_needed()
    net = None
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
        print("Cargando red neuronal...")
        net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        print("Red neuronal cargada exitosamente.")
    else:
        print("ADVERTENCIA: No se encontraron los archivos del modelo DNN.")

    src = select_input_source()
    print(f"\nUsando fuente: {src if isinstance(src, str) else 'Cámara'}\n")

    use_grabber = isinstance(src, int) or (isinstance(src,str) and src.startswith("rtsp"))
    
    if use_grabber:
        try:
            grabber = FrameGrabber(src).start()
            frame_delay = 1
            cap = None
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            print("Posibles soluciones:")
            print("- Verifica que la cámara esté conectada")
            print("- Cierra otras aplicaciones que usen la cámara")
            print("- Intenta con un video en su lugar")
            return
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print("ERROR: No se pudo abrir el video.")
            return
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps > 0:
            frame_delay = int(1000 / video_fps)
            print(f"FPS del video: {video_fps:.2f} | Delay: {frame_delay}ms")
        else:
            frame_delay = 30
        grabber = None

    if use_grabber:
        ret, first_frame = grabber.read()
        forceExit = False
        timeout = 100
        while not ret and not forceExit:
            ret, first_frame = grabber.read()
            time.sleep(0.01)
            timeout -= 1
            if timeout == 0:
                forceExit = True
        if forceExit:
            print("No se pudo leer el primer frame.")
            grabber.stop()
            return
    else:
        ret, first_frame = cap.read()
        if not ret:
            print("No se pudo leer el primer frame del video.")
            cap.release()
            return

    first_frame = cv2.flip(first_frame, 1)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_edges = None

    # Trackeo basado en movimiento
    p0_motion = None
    mask_tracks_motion = np.zeros_like(first_frame)
    movement_history_motion = None

    # Trackeo basado en Canny
    p0_canny = None
    mask_tracks_canny = np.zeros_like(first_frame)
    movement_history_canny = None
    
    frame_count = 0
    # Contornos que consideramos parte del fondo (estáticos)
    # Cada elemento: {"cx", "cy", "w", "h", "life", "last_seen"}
    static_contours = []

    # last_reset_time = time.time()
    # RESET_INTERVAL = 10.0  # segundos
    
    prev_dynamic_bboxes = []
    last_dnn_time = 0.0
    DNN_COOLDOWN_SEC = 0.25   # evita spamear si hay “aparece” en frames consecutivos

    
    #para trackear bounding boxes
    bbox_tracks = []   # cada track: {"id": int, "bbox": (x0,y0,x1,y1), "last_seen": frame_idx}
    next_track_id = 1
    TRACK_IOU_MATCH = 0.30
    TRACK_TTL = 20     # frames sin verse -> borrar



    last_detected_objects = [] # Para guardar detecciones entre frames
    try:
        while True:
            if use_grabber:
                ret, frame = grabber.read()
                if not ret:
                    time.sleep(0.01)
                    print("No se pudo leer un frame.")
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Video terminado.")
                    break

            frame = cv2.flip(frame, 1)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            brightness = calculate_brightness(frame)
            if calibration_mode:
                adaptive_sens, adaptive_blur, adaptive_clip, adaptive_tile = get_calibration_params()
            else:
                adaptive_sens, adaptive_blur, adaptive_clip, adaptive_tile = adjust_canny_params(brightness)

            diff = cv2.absdiff(frame_gray, prev_gray)
            _, motion_mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, None, iterations=DILATE_ITER)

            blur_diff = cv2.GaussianBlur(diff, (3, 3), 0)
            _, edges_diff = cv2.threshold(blur_diff, 25, 255, cv2.THRESH_BINARY)
            edges_diff = cv2.bitwise_and(edges_diff, motion_mask)

            edges = canny_mejorado(
                frame_gray,
                sens=adaptive_sens,
                clip_limit=adaptive_clip,
                tile_size=adaptive_tile,
                blur_kernel=adaptive_blur
            )

            edges = cv2.bitwise_or(edges, edges_diff)

            if prev_edges is not None:
                edge_diff = cv2.absdiff(edges, prev_edges)
                _, moving_edges_mask = cv2.threshold(edge_diff, STATIC_EDGE_THRESHOLD, 255, cv2.THRESH_BINARY)
                edges_moving = cv2.bitwise_and(edges, moving_edges_mask)
            else:
                edges_moving = edges.copy()
            
            prev_edges = edges.copy()

            kernel = np.ones((3, 3), np.uint8)
            edges_thick = cv2.dilate(edges_moving, kernel, iterations=1)

            edges_closed = cv2.morphologyEx(edges_moving, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(edges_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            closed_contours = []
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                    continue
                closed_contours.append(cnt)

            dynamic_contours = update_static_contours(
                closed_contours, static_contours, frame_count
            )

            num_closed = len(dynamic_contours)

            dynamic_mask = np.zeros_like(edges_moving)
            cv2.drawContours(dynamic_mask, dynamic_contours, -1, 255, thickness=1)
            contours_only = cv2.bitwise_and(edges_moving, dynamic_mask)

            dynamic_bboxes = get_dynamic_bounding_boxes(dynamic_contours)
            dynamic_bboxes = merge_bboxes_enclosing(dynamic_bboxes, pad=10, iou_thr=0.15, near_thr=50)

            p0_canny, movement_history_canny = ensure_points_in_bboxes(
                prev_gray=prev_gray,
                mask_img=edges_moving,
                p0=p0_canny,
                movement_hist=movement_history_canny,
                bboxes=[tr["bbox"] for tr in bbox_tracks],
                min_pts_per_bbox=6,
                max_new_per_bbox=25,
                margin=15
            )

            matched_tracks = set()

            for db in dynamic_bboxes:
                best_iou = 0.0
                best_idx = None

                for i, tr in enumerate(bbox_tracks):
                    if i in matched_tracks:
                        continue  # este track ya fue asignado en este frame

                    v = iou(db, tr["bbox"])
                    if v > best_iou:
                        best_iou = v
                        best_idx = i

                if best_idx is not None and best_iou >= TRACK_IOU_MATCH:
                    oldb = bbox_tracks[best_idx]["bbox"]
                    bbox_tracks[best_idx]["bbox"] = bbox_union(oldb, db)   # <- clave: unión
                    bbox_tracks[best_idx]["last_seen"] = frame_count
                    matched_tracks.add(best_idx)
                else:
                    bbox_tracks.append({"id": next_track_id, "bbox": db, "last_seen": frame_count})
                    next_track_id += 1

            # Si una bbox observada cubre múltiples tracks, colapsarlos en uno envolvente
            new_tracks = []
            consumed = set()

            for db in dynamic_bboxes:
                hits = []
                for idx, tr in enumerate(bbox_tracks):
                    if idx in consumed:
                        continue
                    if iou(db, tr["bbox"]) >= 0.20 or check_intersection(db, tr["bbox"]):
                        hits.append((idx, tr))

                if len(hits) <= 1:
                    continue

                # Crear un track único que engloba todos los hits (mantiene el id del primero)
                keep_idx, keep_tr = hits[0]
                xs0 = [t["bbox"][0] for _, t in hits]
                ys0 = [t["bbox"][1] for _, t in hits]
                xs1 = [t["bbox"][2] for _, t in hits]
                ys1 = [t["bbox"][3] for _, t in hits]

                enclosing = (min(xs0), min(ys0), max(xs1), max(ys1))
                keep_tr["bbox"] = bbox_union(keep_tr["bbox"], enclosing)  # <- no shrink
                keep_tr["last_seen"] = frame_count


                for idx, _ in hits[1:]:
                    consumed.add(idx)

            # Remover tracks consumidos
            bbox_tracks = [tr for i, tr in enumerate(bbox_tracks) if i not in consumed]
            bbox_tracks = [tr for tr in bbox_tracks if (frame_count - tr["last_seen"]) <= TRACK_TTL]
            bbox_tracks = merge_tracks(bbox_tracks, iou_thr=0.20, contain_thr=0.90)

            run_dnn = False
            if net is not None:
                now = time.time()
                if (now - last_dnn_time) >= DNN_COOLDOWN_SEC:
                    if appeared_new_region(dynamic_bboxes, prev_dynamic_bboxes, iou_thr=0.20, min_area=800):
                        run_dnn = True
                        last_dnn_time = now

            if run_dnn:
                current_detections = []
                h, w = frame.shape[:2]

                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > DNN_CONFIDENCE:
                        idx = int(detections[0, 0, i, 1])
                        label = DNN_CLASSES[idx]

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        det_bbox = (startX, startY, endX, endY)

                        # Confirmar que intersecta movimiento
                        is_moving = any(check_intersection(det_bbox, m_bbox) for m_bbox in dynamic_bboxes)
                        if is_moving:
                            current_detections.append((label, confidence, det_bbox))

                last_detected_objects = current_detections

            # IMPORTANTE: actualizar “prev” SIEMPRE (corras o no la DNN)
            prev_dynamic_bboxes = list(dynamic_bboxes)

            frame_count += 1
            processed_motion = frame.copy()

            mask_tracks_motion = cv2.addWeighted(mask_tracks_motion, 0.95, mask_tracks_motion, 0, 0)
            
            need_new_features_motion = False
            if p0_motion is None:
                need_new_features_motion = True
            elif len(p0_motion) < MIN_FEATURES and frame_count % 30 == 0:
                need_new_features_motion = True

            if need_new_features_motion:
                p0_motion = cv2.goodFeaturesToTrack(
                    prev_gray,
                    mask=motion_mask,
                    **feature_params
                )
                if p0_motion is not None:
                    movement_history_motion = np.zeros(len(p0_motion), dtype=np.float32)

            if p0_motion is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, frame_gray, p0_motion, None, **lk_params
                )

                if p1 is not None:
                    st = st.reshape(-1)
                    p0_flat = p0_motion.reshape(-1, 2)
                    p1_flat = p1.reshape(-1, 2)

                    displacements = np.linalg.norm(p1_flat - p0_flat, axis=1)

                    if movement_history_motion is None or movement_history_motion.shape[0] != displacements.shape[0]:
                        movement_history_motion = np.zeros_like(displacements)

                    movement_history_motion += displacements
                    inside_bbox = filter_points_by_bboxes(p1, dynamic_bboxes, margin=20)
                    has_moved_before = movement_history_motion >= MIN_MOVEMENT_HISTORY
                    valid = (st == 1) & (displacements < MAX_DISPLACEMENT) & (inside_bbox | has_moved_before)


                    good_new = p1_flat[valid]
                    good_old = p0_flat[valid]

                    if len(good_new) > 0:
                        for (new, old) in zip(good_new, good_old):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            mask_tracks_motion = cv2.line(
                                mask_tracks_motion,
                                (int(a), int(b)), (int(c), int(d)),
                                (0, 255, 0), 2
                            )
                            processed_motion = cv2.circle(
                                processed_motion,
                                (int(a), int(b)),
                                4,
                                (0, 0, 255),
                                -1
                            )

                        p0_motion = good_new.reshape(-1, 1, 2)
                        movement_history_motion = movement_history_motion[valid]
                    else:
                        p0_motion = None
                        movement_history_motion = None
                else:
                    p0_motion = None
                    movement_history_motion = None

            processed_motion = cv2.add(processed_motion, mask_tracks_motion)

            processed_canny = cv2.cvtColor(edges_moving, cv2.COLOR_GRAY2BGR)

            mask_tracks_canny = cv2.addWeighted(mask_tracks_canny, 0.95, mask_tracks_canny, 0, 0)
            need_new_features_canny = False
            if p0_canny is None:
                p0_canny = cv2.goodFeaturesToTrack(prev_gray, mask=edges_moving, **feature_params)
                if p0_canny is not None:
                    movement_history_canny = np.zeros(len(p0_canny), dtype=np.float32)
                else:
                    need_new_features_canny = True

            if need_new_features_canny:
                p0_canny = cv2.goodFeaturesToTrack(
                    prev_gray,
                    mask=edges_moving,
                    **feature_params
                )
                if p0_canny is not None:
                    movement_history_canny = np.zeros(len(p0_canny), dtype=np.float32)

                if (not need_new_features_canny) and (p0_canny is not None) and dynamic_bboxes:
                    points_flat = p0_canny.reshape(-1, 2)

                    for (x_min, y_min, x_max, y_max) in dynamic_bboxes:
                        inside = (
                            (points_flat[:, 0] >= x_min) &
                            (points_flat[:, 0] <= x_max) &
                            (points_flat[:, 1] >= y_min) &
                            (points_flat[:, 1] <= y_max)
                        )

                        if not np.any(inside):
                            roi_mask = np.zeros_like(edges_moving)
                            cv2.rectangle(roi_mask, (x_min, y_min), (x_max, y_max), 255, -1)
                            roi_mask = cv2.bitwise_and(roi_mask, edges_moving)

                            new_pts = cv2.goodFeaturesToTrack(
                                prev_gray,
                                mask=roi_mask,
                                **feature_params
                            )

                            if new_pts is not None and len(new_pts) > 0:
                                if p0_canny is None:
                                    p0_canny = new_pts
                                    points_flat = p0_canny.reshape(-1, 2)
                                else:
                                    p0_canny = np.vstack([p0_canny, new_pts])
                                    points_flat = p0_canny.reshape(-1, 2)


            if p0_canny is not None:
                p1c, stc, errc = cv2.calcOpticalFlowPyrLK(
                    prev_gray, frame_gray, p0_canny, None, **lk_params
                )

                if p1c is not None:
                    stc = stc.reshape(-1)
                    p0c_flat = p0_canny.reshape(-1, 2)
                    p1c_flat = p1c.reshape(-1, 2)

                    disp_c = np.linalg.norm(p1c_flat - p0c_flat, axis=1)

                    # Asegurar que movement_history_canny tenga el mismo tamaño que disp_c
                    if movement_history_canny is None or movement_history_canny.shape[0] != disp_c.shape[0]:
                        movement_history_canny = np.zeros_like(disp_c)

                    movement_history_canny += disp_c

                    
                    # inside_bbox_c = filter_points_by_bbox(p1c, dynamic_bbox, margin=20)
                    inside_bbox_c = filter_points_by_bboxes(p1c, dynamic_bboxes, margin=20)

                    has_moved_before_c = movement_history_canny >= MIN_MOVEMENT_HISTORY
                    # valid_c = (stc == 1) & (disp_c < MAX_DISPLACEMENT) & (inside_bbox_c | has_moved_before_c)
                    valid_c = (stc == 1) & (disp_c < MAX_DISPLACEMENT) & (inside_bbox_c | has_moved_before_c)


                    good_new_c = p1c_flat[valid_c]
                    good_old_c = p0c_flat[valid_c]

                    # --- MOVER BBOX TRACKS CON EL FLUJO OPTICO (CANNY LK) ---
                    if len(good_new_c) > 0 and len(bbox_tracks) > 0:
                        hF, wF = frame.shape[:2]
                        # vectores por punto
                        flows = good_new_c - good_old_c  # (N,2)

                        for tr in bbox_tracks:
                            b = tr["bbox"]

                            # puntos cuyo "old" cae dentro de la bbox
                            idxs = []
                            for k, pold in enumerate(good_old_c):
                                if point_in_bbox(pold, b, margin=5):
                                    idxs.append(k)

                            if len(idxs) >= 3:
                                dx = float(np.median(flows[idxs, 0]))
                                dy = float(np.median(flows[idxs, 1]))
                                moved = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
                                tr["bbox"] = clamp_bbox(moved, wF, hF)
                                tr["last_seen"] = frame_count


                    if len(good_new_c) > 0:
                        for (new, old) in zip(good_new_c, good_old_c):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            mask_tracks_canny = cv2.line(
                                mask_tracks_canny,
                                (int(a), int(b)), (int(c), int(d)),
                                (255, 0, 255), 2
                            )
                            processed_canny = cv2.circle(
                                processed_canny,
                                (int(a), int(b)),
                                4,
                                (0, 255, 255),
                                -1
                            )

                        p0_canny = good_new_c.reshape(-1, 1, 2)
                        movement_history_canny = movement_history_canny[valid_c]
                    else:
                        p0_canny = None
                        movement_history_canny = None
                else:
                    p0_canny = None
                    movement_history_canny = None

            processed_canny = cv2.add(processed_canny, mask_tracks_canny)
            
            for tr in bbox_tracks:
                x0, y0, x1, y1 = tr["bbox"]
                cv2.rectangle(processed_canny, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(processed_canny, f"trk {tr['id']}", (x0, max(0, y0-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(frame, f"trk {tr['id']}", (x0, max(0, y0-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if show_debug_info:
                num_motion = len(p0_motion) if p0_motion is not None else 0
                num_canny = len(p0_canny) if p0_canny is not None else 0
                num_dynamic = len(dynamic_contours)
                
                processed_motion = draw_debug_info(
                    processed_motion, brightness, adaptive_sens, adaptive_blur,
                    adaptive_clip, adaptive_tile, num_motion, num_canny, num_dynamic
                )

            prev_gray = frame_gray.copy()
            edges_bgr = cv2.cvtColor(contours_only, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(edges_bgr, dynamic_contours, -1, (0, 255, 0), 2)

            if dynamic_bboxes:
                for (x_min, y_min, x_max, y_max) in dynamic_bboxes:
                    cv2.rectangle(edges_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            # Dibujar objetos detectados por DNN en edges_bgr
            for (label, conf, (startX, startY, endX, endY)) in last_detected_objects:
                # Rectángulo verde para objetos reconocidos
                cv2.rectangle(edges_bgr, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label_text = f"{label}: {conf:.2f}"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(edges_bgr, label_text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Dibujar objetos detectados por DNN en el FRAME ORIGINAL
            for (label, conf, (startX, startY, endX, endY)) in last_detected_objects:
                # Rectángulo verde para objetos reconocidos en movimiento
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
                label_text = f"{label}: {conf:.2f}"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label_text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


            # Mostrar la cantidad de contornos cerrados EN MOVIMIENTO
            cv2.putText(
                edges_bgr,
                f"Contornos en movimiento: {num_closed}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                edges_bgr,
                f"Brillo: {int(brightness)} | Sens: {adaptive_sens:.2f} | Blur: {adaptive_blur}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

            top_row = np.hstack((frame, processed_motion))
            bottom_row = np.hstack((edges_bgr, processed_canny))
            combined = np.vstack((top_row, bottom_row))

            hC, wC = combined.shape[:2]
            scale_w = TARGET_W / float(wC)
            scale_h = TARGET_H / float(hC)
            scale = min(scale_w, scale_h)
            new_w = max(1, int(wC * scale))
            new_h = max(1, int(hC * scale))
            combined_resized = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.imshow("Arriba: original | LK movimiento  |  Abajo: Canny MEJORADO | LK usando Canny", combined_resized)

            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                mask_tracks_motion = np.zeros_like(frame)
                mask_tracks_canny = np.zeros_like(frame)
                p0_motion = None
                p0_canny = None
                movement_history_canny = None
                movement_history_motion = None
            elif key == ord('w'):
                calibration_mode = not calibration_mode
                if calibration_mode:
                    create_calibration_window()
                    print("Modo calibración ACTIVADO")
                else:
                    cv2.destroyWindow('Calibración Canny')
                    print("Modo calibración DESACTIVADO (modo automático)")
            elif key == ord('d'):
                show_debug_info = not show_debug_info
                if show_debug_info:
                    print("Debug info ACTIVADO")
                else:
                    print("Debug info DESACTIVADO")
    finally:
        if use_grabber and grabber is not None:
            grabber.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
