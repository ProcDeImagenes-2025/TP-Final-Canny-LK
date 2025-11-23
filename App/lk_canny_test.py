import cv2
import numpy as np

# ---------------------------
# Parámetros Lucas-Kanade
# ---------------------------
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Puntos de interés
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Parámetros de detección de movimiento
MOTION_THRESH = 15
MIN_FEATURES = 40
DILATE_ITER = 2

# Filtro de outliers de movimiento
MAX_DISPLACEMENT = 40.0  # píxeles

# Canny
CANNY_LOW = 50
CANNY_HIGH = 150


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

        # 2) Canny
        edges = cv2.Canny(frame_gray, CANNY_LOW, CANNY_HIGH)

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

                    xs = good_new[:, 0]
                    ys = good_new[:, 1]
                    # min_x, max_x = int(np.min(xs)), int(np.max(xs))
                    # min_y, max_y = int(np.min(ys)), int(np.max(ys))
                    # cv2.rectangle(
                    #     processed_motion,
                    #     (min_x, min_y), (max_x, max_y),
                    #     (255, 0, 0), 2
                    # )

                    p0_motion = good_new.reshape(-1, 1, 2)
                else:
                    p0_motion = None
            else:
                p0_motion = None

        processed_motion = cv2.add(processed_motion, mask_tracks_motion)

        # 4) Trackeo usando Canny
        # Esto sólo toma puntos que están en los bordes detectados por Canny, sería bueno tomar los que están dentro de los contornos.
        processed_canny = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # buscamos esquinas SOLO en bordes Canny
        need_new_features_canny = False
        if p0_canny is None:
            need_new_features_canny = True
        elif len(p0_canny) < MIN_FEATURES and frame_count % 10 == 0:
            need_new_features_canny = True

        if need_new_features_canny:
            p0_canny = cv2.goodFeaturesToTrack(
                prev_gray,
                mask=edges,   # máscara = bordes Canny
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
                            (255, 0, 255), 2  # magenta
                        )
                        processed_canny = cv2.circle(
                            processed_canny,
                            (int(a), int(b)),
                            4,
                            (0, 255, 255),  # amarillo
                            -1
                        )

                    xs = good_new_c[:, 0]
                    ys = good_new_c[:, 1]
                    # min_x, max_x = int(np.min(xs)), int(np.max(xs))
                    # min_y, max_y = int(np.min(ys)), int(np.max(ys))
                    # cv2.rectangle(
                    #     processed_canny,
                    #     (min_x, min_y), (max_x, max_y),
                    #     (255, 255, 0), 2  # cian-amarillo
                    # )

                    p0_canny = good_new_c.reshape(-1, 1, 2)
                else:
                    p0_canny = None
            else:
                p0_canny = None

        processed_canny = cv2.add(processed_canny, mask_tracks_canny)

        # 5) Actualizar prev_gray y mostrar
        prev_gray = frame_gray.copy()

        # Pasar edges a BGR para poder apilarlo
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Asegurar mismo tamaño
        h, w = frame.shape[:2]
        processed_motion = cv2.resize(processed_motion, (w, h))
        edges_bgr = cv2.resize(edges_bgr, (w, h))
        processed_canny = cv2.resize(processed_canny, (w, h))

        top_row = np.hstack((frame, processed_motion))
        bottom_row = np.hstack((edges_bgr, processed_canny))
        combined = np.vstack((top_row, bottom_row))

        cv2.imshow("Arriba: original | LK movimiento  |  Abajo: Canny | LK usando Canny", combined)

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
