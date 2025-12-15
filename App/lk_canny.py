import cv2
import numpy as np
import time
import os
import urllib.request
import threading
from utils.frameGrabber import FrameGrabber

# ---------------------------
# Configuración de Pantalla
# ---------------------------
TARGET_W = 1000
TARGET_H = 600

# ---------------------------
# Configuración Red Neuronal
# ---------------------------
DNN_PROTO = "MobileNetSSD_deploy.prototxt"
DNN_MODEL = "MobileNetSSD_deploy.caffemodel"
DNN_CONFIDENCE = 0.5
DNN_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

def download_model_if_needed():
    """Descarga el modelo MobileNet-SSD si no existe localmente"""
    if not os.path.exists(DNN_PROTO):
        print(f"Descargando {DNN_PROTO}...")
        try:
            # URL alternativa que suele ser más estable
            url_proto = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"
            urllib.request.urlretrieve(url_proto, DNN_PROTO)
        except Exception as e:
            print(f"Error descargando prototxt: {e}")
    
    if not os.path.exists(DNN_MODEL):
        print(f"Descargando {DNN_MODEL}...")
        try:
            # URL alternativa que suele ser más estable
            url_model = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel" 
            urllib.request.urlretrieve(url_model, DNN_MODEL)
        except Exception as e:
            print(f"Error descargando caffemodel: {e}")

def merge_rectangles(rects, threshold=40):
    """
    Fusiona rectángulos cercanos o superpuestos.
    rects: lista de tuplas (x_min, y_min, x_max, y_max)
    threshold: distancia máxima en píxeles para considerar fusión
    """
    if not rects:
        return []
    
    # Convertir a lista de listas para poder modificar
    merged = [list(r) for r in rects]
    
    while True:
        new_merged = []
        used = [False] * len(merged)
        changed = False
        
        for i in range(len(merged)):
            if used[i]:
                continue
            
            # Rectángulo base para intentar fusionar
            current = merged[i]
            used[i] = True
            
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                
                other = merged[j]
                
                # Verificar cercanía/superposición con threshold
                # Se tocan si:
                # A.left < B.right + th AND A.right + th > B.left ...
                if (current[0] - threshold < other[2] and
                    current[2] + threshold > other[0] and
                    current[1] - threshold < other[3] and
                    current[3] + threshold > other[1]):
                    
                    # Fusionar: tomar los límites extremos
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
            
    # Convertir de vuelta a tuplas
    return [tuple(r) for r in merged]

def check_intersection(boxA, boxB):
    """
    Verifica si dos rectángulos se intersectan.
    box: (x_min, y_min, x_max, y_max)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calcular área de intersección
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea > 0

# ---------------------------
# Parámetros Lucas-Kanade
# ---------------------------
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Puntos de interés
feature_params = dict(
    maxCorners=300,
    qualityLevel=0.15,
    minDistance=10,
    blockSize=7
)

# Parámetros de detección de movimiento
MOTION_THRESH = 15
MIN_FEATURES = 10
DILATE_ITER = 4


# Filtro de outliers de movimiento
MAX_DISPLACEMENT = 1000.0  # píxeles

# NUEVO: Umbral para considerar que un punto "se movió alguna vez"
MIN_MOVEMENT_HISTORY = 5.0  # píxeles acumulados

# Umbral para considerar que un borde está "quieto"
STATIC_EDGE_THRESHOLD = 3   # Si el borde no cambia más de 5 píxeles, se ignora

# Parámetros Canny mejorado
CANNY_SENS = 0.33
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
BLUR_KERNEL = 5

# Factor de escala para la ventana (0.5 = mitad del tamaño, 0.75 = 75%, etc.)
DISPLAY_SCALE = 0.5

# Área mínima para considerar un contorno "real" (en píxeles)
MIN_CONTOUR_AREA = 100

# Parámetros para filtrar contornos "estáticos" (fondo)
STATIC_CONTOUR_FRAMES = 8      # frames necesarios para marcarlo como estático
STATIC_MATCH_DIST = 10.0       # tolerancia de distancia entre centros (px)
STATIC_SIZE_TOL = 0.4          # tolerancia de tamaño (±40%)
STATIC_FORGET_FRAMES = 60      # si no aparece en 60 frames, se descarta del fondo


# def calculate_brightness(frame):
#     """Calcula el nivel de brillo promedio de la imagen usando el canal V de HSV"""
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     v_channel = hsv[:, :, 2]  # Canal V (Value/Brightness)
#     brightness = np.mean(v_channel)
#     return brightness

calibration_mode = False
show_debug_info = False
manual_sens = int(CANNY_SENS * 100)  # 0-100
manual_clip = int(CLAHE_CLIP_LIMIT * 10)  # 0-100 (dividir por 10)
manual_tile = CLAHE_TILE_SIZE  # 2-16
manual_blur = BLUR_KERNEL  # 1-31 (impares)


def get_dynamic_bounding_box(dynamic_contours):
    """
    Crea una bounding box que engloba TODOS los contornos dinámicos
    Retorna: (x_min, y_min, x_max, y_max) o None si no hay contornos
    """
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
    """
    Crea una lista de bounding boxes, una por cada contorno dinámico.
    Cada bbox es (x_min, y_min, x_max, y_max).
    Si no hay contornos, devuelve lista vacía.
    """
    bboxes = []
    for cnt in dynamic_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((int(x), int(y), int(x + w), int(y + h)))
    return bboxes

def filter_points_by_bbox(points, bbox, margin=10):
    """
    Filtra puntos que están dentro de la bounding box (con margen opcional)
    
    Args:
        points: array de puntos (N, 1, 2)
        bbox: (x_min, y_min, x_max, y_max)
        margin: píxeles extra alrededor de la bbox
    
    Returns:
        array de índices válidos
    """
    if points is None or len(points) == 0:
        return np.array([], dtype=bool)
    
    # Si no hay bbox, marcar todos los puntos como inválidos
    if bbox is None:
        return np.zeros(len(points), dtype=bool)
    
    x_min, y_min, x_max, y_max = bbox
    
    # Expandir bbox con margen
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
    """
    Filtra puntos que están dentro de AL MENOS una bounding box (con margen opcional).
    
    Args:
        points: array de puntos (N, 1, 2)
        bboxes: lista de (x_min, y_min, x_max, y_max)
        margin: píxeles extra alrededor de cada bbox
    
    Returns:
        array de booleanos de tamaño N indicando qué puntos son válidos
    """
    if points is None or len(points) == 0:
        return np.array([], dtype=bool)
    
    if not bboxes:  # lista vacía
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
    """Crea ventana de calibración con trackbars"""
    cv2.namedWindow('Calibración Canny')
    cv2.createTrackbar('Sensibilidad x100', 'Calibración Canny', manual_sens, 100, nothing)
    cv2.createTrackbar('CLAHE Clip x10', 'Calibración Canny', manual_clip, 100, nothing)
    cv2.createTrackbar('CLAHE Tile', 'Calibración Canny', manual_tile, 16, nothing)
    cv2.createTrackbar('Blur Kernel', 'Calibración Canny', manual_blur, 31, nothing)
    
    # Crear imagen informativa
    info = np.zeros((200, 500, 3), dtype=np.uint8)
    cv2.putText(info, 'Ajusta los parametros:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info, 'Sensibilidad: 0-100 (x0.01)', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(info, 'CLAHE Clip: 0-100 (x0.1)', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(info, 'CLAHE Tile: 2-16', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(info, 'Blur Kernel: 1-31 (impar)', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow('Calibración Canny', info)

def draw_debug_info(frame, brightness, sens, blur, clip, tile, num_points_motion, num_points_canny, num_contours):
    """Dibuja información de debug en el frame"""
    # Fondo semi-transparente para el texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
    frame_with_info = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    y_offset = 35
    line_height = 30
    
    # Título
    cv2.putText(frame_with_info, "DEBUG INFO (tecla 'd' para ocultar)", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += line_height
    
    # Parámetros Canny
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
    
    # Tracking info
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
    """Obtiene los valores actuales de los trackbars"""
    global manual_sens, manual_clip, manual_tile, manual_blur
    
    manual_sens = cv2.getTrackbarPos('Sensibilidad x100', 'Calibración Canny')
    manual_clip = cv2.getTrackbarPos('CLAHE Clip x10', 'Calibración Canny')
    manual_tile = max(2, cv2.getTrackbarPos('CLAHE Tile', 'Calibración Canny'))
    manual_blur = cv2.getTrackbarPos('Blur Kernel', 'Calibración Canny')
    
    # Asegurar que blur_kernel sea impar
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
    """
    Ajusta parámetros Canny con ecuaciones continuas basadas en brillo (0-170)
    
    Ecuaciones interpoladas entre:
    - Brillo bajo (0-80): sens=0.33, blur=19, clip=4.0, tile=8
    - Brillo medio (80-170): sens=0.5, blur=23, clip=3.0, tile=4
    
    Brightness range usado: 0-170
    """
    # Limitar brillo al rango calibrado
    brightness = min(brightness, 170)
    
    # Factor de interpolación normalizado (0.0 a 1.0)
    t = brightness / 170.0
    
    # 1. Sensibilidad: aumenta linealmente de 0.33 a 0.5
    #    A más luz, mayor sensibilidad para capturar bordes sutiles
    sens = 0.33 + t * 0.17
    
    # 2. Blur kernel: aumenta de 19 a 23
    #    A más luz, más suavizado para reducir ruido y texturas
    blur_float = 19.0 + t * 4.0
    blur = int(round(blur_float))
    # Asegurar que sea impar
    if blur % 2 == 0:
        blur += 1
    blur = max(1, min(31, blur))  # Limitar a rango válido
    
    # 3. CLAHE Clip: decrece de 4.0 a 3.0
    #    Menos realce de contraste en escenas iluminadas
    clip = 4.0 - t * 1.0
    
    # 4. CLAHE Tile: decrece de 8 a 4
    #    Tiles más pequeños para mejor adaptación local con luz
    tile_float = 8.0 - t * 4.0
    tile = int(round(tile_float))
    tile = max(2, min(16, tile))  # Limitar a rango válido
    
    return sens, blur, clip, tile


def canny_mejorado(img, sens=0.33, clip_limit=2.0, tile_size=8, blur_kernel=5):
    """Canny con CLAHE + Blur para mejor detección de bordes"""
    # 1. CLAHE para ecualizar
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    equalized = clahe.apply(img)
    
    # 2. Blur para reducir ruido/texturas
    blurred = cv2.GaussianBlur(equalized, (blur_kernel, blur_kernel), 0)
    
    # 3. Canny con thresholds automáticos
    med = np.median(blurred)
    low_thresh = int(max(0, (1.0 - sens) * med))
    high_thresh = int(min(255, (1.0 + sens) * med))
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    return edges

def update_static_contours(closed_contours, static_contours, frame_idx):
    """
    Actualiza la lista de contornos estáticos y devuelve
    sólo los contornos que NO son estáticos (dinámicos).
    """
    dynamic_contours = []

    for cnt in closed_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0

        best_idx = None
        best_dist = 1e9

        # Buscar si este contorno coincide con alguno ya visto
        for i, sc in enumerate(static_contours):
            dist = np.hypot(cx - sc["cx"], cy - sc["cy"])
            if dist > STATIC_MATCH_DIST:
                continue

            # Comparar tamaño (evitar matchear cosas muy distintas)
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
            # Actualizar contorno existente
            sc = static_contours[best_idx]
            sc["cx"] = 0.5 * (sc["cx"] + cx)
            sc["cy"] = 0.5 * (sc["cy"] + cy)
            sc["w"] = 0.5 * (sc["w"] + w)
            sc["h"] = 0.5 * (sc["h"] + h)
            sc["life"] += 1
            sc["last_seen"] = frame_idx

            is_static = sc["life"] >= STATIC_CONTOUR_FRAMES
        else:
            # Contorno nuevo
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

        # Sólo nos quedamos con los contornos que TODAVÍA no
        # fueron clasificados como estáticos
        if not is_static:
            dynamic_contours.append(cnt)

    # Olvidar contornos que hace mucho que no aparecen
    static_contours[:] = [
        sc
        for sc in static_contours
        if frame_idx - sc["last_seen"] <= STATIC_FORGET_FRAMES
    ]

    return dynamic_contours



def main():
    global calibration_mode, show_debug_info
    
    # Descargar y cargar modelo DNN
    download_model_if_needed()
    net = None
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
        print("Cargando red neuronal...")
        net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        print("Red neuronal cargada exitosamente.")
    else:
        print("ADVERTENCIA: No se encontraron los archivos del modelo DNN.")

    #cap = cv2.VideoCapture("rtsp://admin:12345678Ab@192.168.1.11:554/1/1?transport=tcp")
    src = "rtsp://admin:12345678Ab@192.168.1.11:554/1/1?transport=tcp"

    grabber = FrameGrabber(src).start()

    #Primer frame
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

    # Espejar para que refleje movimientos naturales
    first_frame = cv2.flip(first_frame, 1)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_edges = None  # Para almacenar bordes del frame anterior

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

    last_reset_time = time.time()
    RESET_INTERVAL = 10.0  # segundos
    
    last_detected_objects = [] # Para guardar detecciones entre frames
    try:
        while True:
            ret, frame = grabber.read()
            if not ret:
                time.sleep(0.01)
                print("No se pudo leer un frame.")
                continue

            # Espejar
            frame = cv2.flip(frame, 1)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            current_time = time.time()
            if current_time - last_reset_time >= RESET_INTERVAL:
                # Resetear puntos y máscaras
                mask_tracks_motion = np.zeros_like(frame)
                mask_tracks_canny = np.zeros_like(frame)
                p0_motion = None
                p0_canny = None
                movement_history_canny = None
                movement_history_motion = None
                last_reset_time = current_time
                print("Puntos reseteados automáticamente")


            # NUEVO: Calcular brillo y ajustar parámetros automáticamente
            brightness = calculate_brightness(frame)
            if calibration_mode:
                adaptive_sens, adaptive_blur, adaptive_clip, adaptive_tile = get_calibration_params()
            else:
                adaptive_sens, adaptive_blur, adaptive_clip, adaptive_tile = adjust_canny_params(brightness)

            # 1) Detección de movimiento global
            diff = cv2.absdiff(frame_gray, prev_gray)
            _, motion_mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, None, iterations=DILATE_ITER)

            # 1.1) Bordes "rápidos" usando solo la diferencia temporal
            blur_diff = cv2.GaussianBlur(diff, (3, 3), 0)
            # Umbral medio-alto para quedarnos sólo con cambios fuertes
            _, edges_diff = cv2.threshold(blur_diff, 25, 255, cv2.THRESH_BINARY)
            edges_diff = cv2.bitwise_and(edges_diff, motion_mask)


            # 2) Canny MEJORADO con parámetros adaptativos
            edges = canny_mejorado(
                frame_gray,
                sens=adaptive_sens,
                clip_limit=adaptive_clip,
                tile_size=adaptive_tile,
                blur_kernel=adaptive_blur
            )

            # 2.1) Combinar bordes espaciales + bordes temporales
            edges = cv2.bitwise_or(edges, edges_diff)

            # 2.2) ELIMINAR BORDES ESTÁTICOS
            if prev_edges is not None:
                # Diferencia entre bordes actuales y anteriores
                edge_diff = cv2.absdiff(edges, prev_edges)
                
                # Solo mantener bordes que cambiaron
                _, moving_edges_mask = cv2.threshold(edge_diff, STATIC_EDGE_THRESHOLD, 255, cv2.THRESH_BINARY)
                
                # Combinar: bordes actuales AND bordes que se movieron
                edges_moving = cv2.bitwise_and(edges, moving_edges_mask)
            else:
                edges_moving = edges.copy()
            
            # Actualizar bordes previos
            prev_edges = edges.copy()

            kernel = np.ones((3, 3), np.uint8)
            edges_thick = cv2.dilate(edges_moving, kernel, iterations=1)

            # 2.2) Contar contornos cerrados solo en bordes en movimiento
            edges_closed = cv2.morphologyEx(edges_moving, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(edges_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            closed_contours = []
            for cnt in contours:
                # descartar contornos muy chicos (ruido)
                if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                    continue
                closed_contours.append(cnt)

            # Filtrar contornos que son "estáticos" (aparecen siempre en el mismo lugar)
            dynamic_contours = update_static_contours(
                closed_contours, static_contours, frame_count
            )

            num_closed = len(dynamic_contours)

            # --- Bordes tipo Canny que sobreviven al filtro de "estático" ---
            # 1) Máscara con las zonas de contornos dinámicos
            dynamic_mask = np.zeros_like(edges_moving)
            cv2.drawContours(dynamic_mask, dynamic_contours, -1, 255, thickness=1)

            # 2) Nos quedamos sólo con los bordes en movimiento que caen dentro
            #    de esos contornos dinámicos
            contours_only = cv2.bitwise_and(edges_moving, dynamic_mask)

            dynamic_bboxes = get_dynamic_bounding_boxes(dynamic_contours)

            # --- FUSIONAR RECTÁNGULOS ---
            # Unir rectángulos cercanos para tener un bounding box más grande y estable
            merged_bboxes = merge_rectangles(dynamic_bboxes, threshold=50)
            dynamic_bboxes = merged_bboxes  # Usamos los fusionados para el resto del pipeline

            # --- DETECCIÓN DE OBJETOS (DNN) ---
            if net is not None and frame_count % 5 == 0:
                current_detections = []
                h, w = frame.shape[:2]
                # Preprocesamiento para MobileNet-SSD
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > DNN_CONFIDENCE:
                        idx = int(detections[0, 0, i, 1])
                        label = DNN_CLASSES[idx]
                        
                        # Coordenadas
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        det_bbox = (startX, startY, endX, endY)
                        
                        # Verificar intersección con movimiento
                        is_moving = False
                        for m_bbox in dynamic_bboxes:
                            if check_intersection(det_bbox, m_bbox):
                                is_moving = True
                                break
                        
                        if is_moving:
                            current_detections.append((label, confidence, det_bbox))
                
                last_detected_objects = current_detections


            # Crear imagen negra solo con contornos cerrados EN MOVIMIENTO (no estáticos)
            # contours_only = np.zeros_like(frame_gray)
            # cv2.drawContours(contours_only, dynamic_contours, -1, (255), 2)
            # contours_only = edges_moving.copy()

            # 3) Trackeo usando movimiento
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

                    # Asegurar que movement_history_motion tenga el mismo tamaño que displacements
                    if movement_history_motion is None or movement_history_motion.shape[0] != displacements.shape[0]:
                        movement_history_motion = np.zeros_like(displacements)

                    movement_history_motion += displacements

                     
                    # inside_bbox = filter_points_by_bbox(p1, dynamic_bbox, margin=20)
                    inside_bbox = filter_points_by_bboxes(p1, dynamic_bboxes, margin=20)

                    # NUEVO: Un punto sobrevive si:
                    # 1. Status válido (st == 1)
                    # 2. Desplazamiento razonable (< MAX_DISPLACEMENT)
                    # 3. Está dentro de bbox O ya se movió antes (tiene historial)
                    has_moved_before = movement_history_motion >= MIN_MOVEMENT_HISTORY
                    # valid = (st == 1) & (displacements < MAX_DISPLACEMENT) & (inside_bbox | has_moved_before)
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

            # 4) Trackeo usando Canny mejorado (solo bordes en movimiento)
            processed_canny = cv2.cvtColor(edges_moving, cv2.COLOR_GRAY2BGR)


            mask_tracks_canny = cv2.addWeighted(mask_tracks_canny, 0.95, mask_tracks_canny, 0, 0)
            need_new_features_canny = False
            if p0_canny is None:
                need_new_features_canny = True
            elif len(p0_canny) < MIN_FEATURES and frame_count % 10 == 0:
                need_new_features_canny = True

            if need_new_features_canny:
                p0_canny = cv2.goodFeaturesToTrack(
                    prev_gray,
                    mask=edges_moving,  # Solo detectar en bordes que se mueven
                    **feature_params
                )
                if p0_canny is not None:
                    movement_history_canny = np.zeros(len(p0_canny), dtype=np.float32)

                # ---------------------------------------------------------------------
                # Agregar nuevos puntos en bboxes que no tienen ninguno (solo Canny)
                # ---------------------------------------------------------------------
                if (not need_new_features_canny) and (p0_canny is not None) and dynamic_bboxes:
                    points_flat = p0_canny.reshape(-1, 2)

                    for (x_min, y_min, x_max, y_max) in dynamic_bboxes:

                        # Chequear si EXISTE al menos un punto dentro de esta bbox
                        inside = (
                            (points_flat[:, 0] >= x_min) &
                            (points_flat[:, 0] <= x_max) &
                            (points_flat[:, 1] >= y_min) &
                            (points_flat[:, 1] <= y_max)
                        )

                        if not np.any(inside):
                            # ---------------------------------------------------------
                            # NO hay puntos en esta bbox → crear máscara y buscar nuevos
                            # ---------------------------------------------------------
                            roi_mask = np.zeros_like(edges_moving)
                            cv2.rectangle(roi_mask, (x_min, y_min), (x_max, y_max), 255, -1)

                            # Restringimos la ROI a bordes que se mueven
                            roi_mask = cv2.bitwise_and(roi_mask, edges_moving)

                            new_pts = cv2.goodFeaturesToTrack(
                                prev_gray,
                                mask=roi_mask,
                                **feature_params
                            )

                            # Si encontramos nuevos puntos, los agregamos al conjunto actual
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

            # 5) Mostrar info de debug si está activado
            if show_debug_info:
                num_motion = len(p0_motion) if p0_motion is not None else 0
                num_canny = len(p0_canny) if p0_canny is not None else 0
                num_dynamic = len(dynamic_contours)
                
                processed_motion = draw_debug_info(
                    processed_motion, brightness, adaptive_sens, adaptive_blur,
                    adaptive_clip, adaptive_tile, num_motion, num_canny, num_dynamic
                )

            # 6) Actualizar prev_gray y mostrar
            prev_gray = frame_gray.copy()

            # Pasar edges a BGR para poder apilarlo
            edges_bgr = cv2.cvtColor(contours_only, cv2.COLOR_GRAY2BGR)

            #############   MOSTRAR CANTIDAD DE CONTRORNOS CERRADOS DETECTADOS  #############
            # Dibujar opcionalmente los contornos cerrados
            cv2.drawContours(edges_bgr, dynamic_contours, -1, (0, 255, 0), 2)


            # if dynamic_bbox is not None:
            #     x_min, y_min, x_max, y_max = dynamic_bbox
            #     cv2.rectangle(edges_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            #     cv2.rectangle(processed_canny, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            if dynamic_bboxes:
                for (x_min, y_min, x_max, y_max) in dynamic_bboxes:
                    cv2.rectangle(edges_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    # cv2.rectangle(processed_canny, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            # Dibujar objetos detectados por DNN
            for (label, conf, (startX, startY, endX, endY)) in last_detected_objects:
                # Rectángulo verde para objetos reconocidos
                cv2.rectangle(edges_bgr, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label_text = f"{label}: {conf:.2f}"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(edges_bgr, label_text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


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

            # Mostrar información de iluminación y parámetros adaptativos
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


            # # Asegurar mismo tamaño
            # h, w = frame.shape[:2]
            # processed_motion = cv2.resize(processed_motion, (w, h))
            # edges_bgr = cv2.resize(edges_bgr, (w, h))
            # processed_canny = cv2.resize(processed_canny, (w, h))

            top_row = np.hstack((frame, processed_motion))
            bottom_row = np.hstack((edges_bgr, processed_canny))
            combined = np.vstack((top_row, bottom_row))

            # -----------------------
            # AUTO-FIT A resolución deseada
            # -----------------------
            hC, wC = combined.shape[:2]

            scale_w = TARGET_W / float(wC)
            scale_h = TARGET_H / float(hC)
            scale = min(scale_w, scale_h)  # mantiene aspect ratio y entra completo

            new_w = max(1, int(wC * scale))
            new_h = max(1, int(hC * scale))

            combined_resized = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)


            cv2.imshow("Arriba: original | LK movimiento  |  Abajo: Canny MEJORADO | LK usando Canny", combined_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                # limpiar líneas y puntos acumulados
                mask_tracks_motion = np.zeros_like(frame)
                mask_tracks_canny = np.zeros_like(frame)
                p0_motion = None
                p0_canny = None
                movement_history_canny = None
                movement_history_motion = None
            elif key == ord('w'):
                # NUEVO: Alternar modo calibración
                calibration_mode = not calibration_mode
                if calibration_mode:
                    create_calibration_window()
                    print("Modo calibración ACTIVADO")
                else:
                    cv2.destroyWindow('Calibración Canny')
                    print("Modo calibración DESACTIVADO (modo automático)")
            elif key == ord('d'):
                # NUEVO: Alternar visualización de debug
                show_debug_info = not show_debug_info
                if show_debug_info:
                    print("Debug info ACTIVADO")
                else:
                    print("Debug info DESACTIVADO")
    finally:
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
