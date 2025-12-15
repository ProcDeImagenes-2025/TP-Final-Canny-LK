import cv2
import threading

class FrameGrabber:
    """
    Hilo dedicado a leer frames continuamente y quedarse siempre con el último.
    Evita backlog/buffer cuando el procesamiento es lento.
    """
    def __init__(self, src, api_preference=cv2.CAP_FFMPEG):
        self.src = src
        self.api_preference = api_preference
        self.cap = None

        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._frame = None
        self._ok = False

    def start(self):
        self.cap = cv2.VideoCapture(self.src, self.api_preference)

        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara (VideoCapture).")

        # Importante en RTSP: reduce buffering (no siempre aplica según backend)
        #try:
        #    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        #except Exception:
        #    pass

        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                # Si la cámara corta, marcamos como inválido y seguimos intentando
                with self._lock:
                    self._ok = False
                    self._frame = None
                # Pequeña pausa para no quemar CPU si la cámara está caída
                time.sleep(0.01)
                continue

            with self._lock:
                self._ok = True
                self._frame = frame  # siempre pisa el frame anterior

    def read(self):
        """
        Devuelve (ok, frame_copy). frame_copy es una copia para evitar data races.
        """
        with self._lock:
            ok = self._ok
            if not ok or self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

