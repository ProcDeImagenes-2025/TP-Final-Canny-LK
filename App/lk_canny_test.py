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
MIN_FEATURES = 50
DILATE_ITER = 4

# Filtro de outliers de movimiento
MAX_DISPLACEMENT = 100.0  # píxeles

# Umbral para considerar que un borde está "quieto"
STATIC_EDGE_THRESHOLD = 1  # Si el borde no cambia más de 5 píxeles, se ignora

# Parámetros Canny mejorado
CANNY_SENS = 0.33
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
BLUR_KERNEL = 5

# Factor de escala para la ventana (0.5 = mitad del tamaño, 0.75 = 75%, etc.)
DISPLAY_SCALE = 0.5

# Área mínima para considerar un contorno "real" (en píxeles)
MIN_CONTOUR_AREA = 100


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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame.")
            break

        # Espejar
        frame = cv2.flip(frame, 1)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) Detección de movimiento global
        diff = cv2.absdiff(frame_gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, None, iterations=DILATE_ITER)

        # 2) Canny MEJORADO
        edges = canny_mejorado(
            frame_gray,
            sens=CANNY_SENS,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_size=CLAHE_TILE_SIZE,
            blur_kernel=BLUR_KERNEL
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

        num_closed = len(closed_contours)

        # Crear imagen negra solo con contornos cerrados EN MOVIMIENTO
        contours_only = np.zeros_like(frame_gray)
        cv2.drawContours(contours_only, closed_contours, -1, (255), 2)

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
        cv2.drawContours(edges_bgr, closed_contours, -1, (0, 255, 0), 2)

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