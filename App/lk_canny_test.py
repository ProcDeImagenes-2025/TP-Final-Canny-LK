import cv2
import numpy as np

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
MIN_FEATURES = 200
DILATE_ITER = 4

# Filtro de outliers de movimiento
MAX_DISPLACEMENT = 100.0  # píxeles

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


def calculate_brightness(frame):
    """Calcula el nivel de brillo promedio de la imagen usando el canal V de HSV"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # Canal V (Value/Brightness)
    brightness = np.mean(v_channel)
    return brightness


def adjust_canny_params(brightness):
    """
    Ajusta CANNY_SENS y BLUR_KERNEL según el nivel de iluminación
    
    Brightness range: 0-255
    - Oscuro (0-80): Más sensibilidad, menos blur
    - Medio (80-170): Valores balanceados
    - Claro (170-255): Menos sensibilidad, más blur
    """
    if brightness < 80:  # Escena oscura
        sens = 0.40  # Más sensible para detectar bordes débiles
        blur = 7     # Menos blur para preservar detalles
        clip = 5.0   # CLAHE muy alto para compensar oscuridad
        tile = 2     # Muy local
    elif brightness < 170:  # Escena con iluminación media
        sens = 0.5
        blur = 17
        clip = 3
        tile = 8
    else:  # Escena muy iluminada
        sens = 0.25  # Menos sensible (evita ruido/texturas)
        blur = 1     # Más blur para suavizar
        clip = 1.5   # CLAHE bajo (ya hay buen contraste)
        tile = 8     # Más global
    
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
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    # Primer frame
    ret, first_frame = cap.read()
    if not ret:
        print("No se pudo leer el primer frame.")
        cap.release()
        return

    # Espejar para que refleje movimientos naturales
    first_frame = cv2.flip(first_frame, 1)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_edges = None  # Para almacenar bordes del frame anterior

    # Trackeo basado en movimiento
    p0_motion = None
    mask_tracks_motion = np.zeros_like(first_frame)

    # Trackeo basado en Canny
    p0_canny = None
    mask_tracks_canny = np.zeros_like(first_frame)

    frame_count = 0
    # Contornos que consideramos parte del fondo (estáticos)
    # Cada elemento: {"cx", "cy", "w", "h", "life", "last_seen"}
    static_contours = []


    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame.")
            break

        # Espejar
        frame = cv2.flip(frame, 1)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # NUEVO: Calcular brillo y ajustar parámetros automáticamente
        brightness = calculate_brightness(frame)
        adaptive_sens, adaptive_blur, adaptive_clip, adaptive_tile = adjust_canny_params(brightness)

        # 1) Detección de movimiento global
        diff = cv2.absdiff(frame_gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, None, iterations=DILATE_ITER)

        # 2) Canny MEJORADO con parámetros adaptativos
        edges = canny_mejorado(
            frame_gray,
            sens=adaptive_sens,
            clip_limit=adaptive_clip,
            tile_size=adaptive_tile,
            blur_kernel=adaptive_blur
        )

        # 2.1) ELIMINAR BORDES ESTÁTICOS
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


        # Crear imagen negra solo con contornos cerrados EN MOVIMIENTO (no estáticos)
        # contours_only = np.zeros_like(frame_gray)
        # cv2.drawContours(contours_only, dynamic_contours, -1, (255), 2)
        contours_only = edges_moving.copy()

        # 3) Trackeo usando movimiento
        frame_count += 1
        processed_motion = frame.copy()

        need_new_features_motion = False
        if p0_motion is None:
            need_new_features_motion = True
        elif len(p0_motion) < MIN_FEATURES and frame_count % 10 == 0:
            need_new_features_motion = True

        if need_new_features_motion:
            p0_motion = cv2.goodFeaturesToTrack(
                prev_gray,
                mask=motion_mask,
                **feature_params
            )

        if p0_motion is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray, p0_motion, None, **lk_params
            )

            if p1 is not None:
                st = st.reshape(-1)
                p0_flat = p0_motion.reshape(-1, 2)
                p1_flat = p1.reshape(-1, 2)

                displacements = np.linalg.norm(p1_flat - p0_flat, axis=1)
                valid = (st == 1) & (displacements < MAX_DISPLACEMENT)

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
                else:
                    p0_motion = None
            else:
                p0_motion = None

        processed_motion = cv2.add(processed_motion, mask_tracks_motion)

        # 4) Trackeo usando Canny mejorado (solo bordes en movimiento)
        processed_canny = cv2.cvtColor(edges_moving, cv2.COLOR_GRAY2BGR)

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
            p1c, stc, errc = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray, p0_canny, None, **lk_params
            )

            if p1c is not None:
                stc = stc.reshape(-1)
                p0c_flat = p0_canny.reshape(-1, 2)
                p1c_flat = p1c.reshape(-1, 2)

                disp_c = np.linalg.norm(p1c_flat - p0c_flat, axis=1)
                valid_c = (stc == 1) & (disp_c < MAX_DISPLACEMENT)

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
                else:
                    p0_canny = None
            else:
                p0_canny = None

        processed_canny = cv2.add(processed_canny, mask_tracks_canny)

        # 5) Actualizar prev_gray y mostrar
        prev_gray = frame_gray.copy()

        # Pasar edges a BGR para poder apilarlo
        edges_bgr = cv2.cvtColor(contours_only, cv2.COLOR_GRAY2BGR)

        #############   MOSTRAR CANTIDAD DE CONTRORNOS CERRADOS DETECTADOS  #############
        # Dibujar opcionalmente los contornos cerrados
        cv2.drawContours(edges_bgr, dynamic_contours, -1, (0, 255, 0), 2)


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


        # Asegurar mismo tamaño
        h, w = frame.shape[:2]
        processed_motion = cv2.resize(processed_motion, (w, h))
        edges_bgr = cv2.resize(edges_bgr, (w, h))
        processed_canny = cv2.resize(processed_canny, (w, h))

        top_row = np.hstack((frame, processed_motion))
        bottom_row = np.hstack((edges_bgr, processed_canny))
        combined = np.vstack((top_row, bottom_row))

        # Redimensionar la ventana completa
        display_h = int(combined.shape[0] * DISPLAY_SCALE)
        display_w = int(combined.shape[1] * DISPLAY_SCALE)
        combined_resized = cv2.resize(combined, (display_w, display_h))

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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()