import cv2
import numpy as np

# ---------------------------
# Parámetros de Lucas-Kanade
# ---------------------------
lk_params = dict(
    winSize=(21, 21),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

# Puntos de interés (features) en zonas con movimiento
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.2,
    minDistance=20,
    blockSize=7
)

# Parámetros de detección de movimiento
MOTION_THRESH = 25      # Umbral sobre la diferencia de intensidad
MIN_FEATURES = 5       # Si hay menos puntos, volvemos a detectar
DILATE_ITER = 2         # Cuántas veces dilatamos la máscara de movimiento
MAX_DISPLACEMENT = 100.0  # en píxeles, para ignorar puntos que salen volando

########################## Prueba CLAHE
# CLAHE para robustez al contraste
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
##########################

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    # Leer primer frame
    ret, first_frame = cap.read()
    if not ret:
        print("No se pudo leer el primer frame de la cámara.")
        cap.release()
        return
    first_frame = cv2.flip(first_frame, 1)  # Flip horizontal para reflejar movimientos naturales

    prev_gray_raw = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = clahe.apply(prev_gray_raw)   # <-- ecualizado

    # No tenemos puntos al principio
    p0 = None

    # Para dibujar trayectorias
    mask_tracks = np.zeros_like(first_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame de la cámara.")
            break
        frame = cv2.flip(frame, 1)  # Para que esté flippeada para que refleje movimientos naturales.

        frame_gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = clahe.apply(frame_gray_raw)

        #Detección de movimiento global
        diff = cv2.absdiff(frame_gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)

        # Limpiar ruido y expandir zonas de movimiento
        motion_mask = cv2.dilate(motion_mask, None, iterations=DILATE_ITER)

        #Detectar features donde hay movimiento
        need_new_features = False
        if p0 is None:
            need_new_features = True
        else:
            # Si se nos murieron casi todos los puntos, busca nuevos
            if len(p0) < MIN_FEATURES:
                need_new_features = True

        if need_new_features:
            p0 = cv2.goodFeaturesToTrack(
                prev_gray,
                mask=motion_mask,
                **feature_params
            )

        processed = frame.copy()

        #Lucas-Kanade: trackeo de puntos
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray, p0, None, **lk_params
            )

            if p1 is not None:
                #puntos con status válido
                st = st.reshape(-1)
                p0_flat = p0.reshape(-1, 2)
                p1_flat = p1.reshape(-1, 2)

                #desplazamiento euclídeo entre frames
                displacements = np.linalg.norm(p1_flat - p0_flat, axis=1)

                #máscara: puntos válidos y que no "vuelan"
                valid = (st == 1) & (displacements < MAX_DISPLACEMENT)

                good_new = p1_flat[valid]
                good_old = p0_flat[valid]

                if len(good_new) > 0:
                    # Dibujar trayectorias y puntos
                    for (new, old) in zip(good_new, good_old):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        mask_tracks = cv2.line(
                            mask_tracks,
                            (int(a), int(b)),
                            (int(c), int(d)),
                            (0, 255, 0),
                            2
                        )
                        processed = cv2.circle(
                            processed,
                            (int(a), int(b)),
                            4,
                            (0, 0, 255),
                            -1
                        )

                    # Combinar tracks con el frame
                    processed = cv2.add(processed, mask_tracks)

                    # Guardar puntos para el próximo frame
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    p0 = None  # No quedaron puntos válidos
            else:
                p0 = None  # Error en LK, volvemos a detectar


        # 4) Actualizar "anterior" y mostrar
        prev_gray = frame_gray.copy()

        # Mostrar original + procesado lado a lado
        combined = np.hstack((frame, processed))
        cv2.imshow("Izquierda: original | Derecha: trackeo movimiento (Lucas-Kanade)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            mask_tracks = np.zeros_like(frame)  #REINICIA LAS LÍNEAS


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
