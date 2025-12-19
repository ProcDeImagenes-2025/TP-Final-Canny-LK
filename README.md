# TP-Final-Canny-LK

# Trabajo final de la materia Procesamiento de Imágenes - ITBA
## Integrantes:
- Hertter, José Iván - 62499
- Oms, Mariano Alejandro - 62080

---

## Nota sobre el estado del documento

El contenido a continuación corresponde a la **primera versión** del proyecto (la presentada en la **primera presentación** del trabajo práctico).  
Se conserva deliberadamente como registro del enfoque inicial y del razonamiento de diseño en esa etapa.

Al final del documento se incluye una sección nueva titulada **“Versión actual del proyecto (presentación final)”**, donde se describen las mejoras y ampliaciones incorporadas en el código actual respecto de esta primera versión.

---

## Descripción
Implementación de un sistema de seguimiento de objetos en video utilizando los algoritmos de detección de bordes de Canny y el método de Lucas-Kanade para el seguimiento de puntos característicos.
La idea principal es detectar features en el primer frame para hacer el trackeo LK y despreciar todos aquellos features que no estén cerca de los bordes detectados por Canny que no sean considerados estáticos.

## Flujo del programa
### Bibliotecas utilizadas
- OpenCV
- NumPy

### Pasos principales
- Selección de Input de video o cámara. Optamos por la selección de la cámara por defecto (0).
- Captura de frame y conversión a escala de grises.
- Obtención del brillo del frame estimando con el valor medio de los pixeles.
- Ajuste de los parámetros de Canny en función del brillo.
- Detección de bordes utilizando el algoritmo de Canny.
    - Obtención de la diferencia entre frames consecutivos para subsanar efectos de motion blur.
    - Dilatación de la imagen de bordes para mejorar la detección.
    - En el bloque de código siguiente se muestra cómo se calcula la máscara de movimiento:
        ```
        diff = abs(frame_gray - prev_gray) 
        _, motion_mask = cv2.threshold(diff, MOTION_THRESH, 255, ...)
        ```
        Acá, el resultado son píxeles blancos solo donde hubo cambios muy fuertes.
- Obtención de features utilizando el método de Shi-Tomasi.
- Filtrado de features cercanos a los bordes detectados.
- Seguimiento de los features utilizando el método de Lucas-Kanade.
- Visualización de los resultados en tiempo real.

### ¿Qué observamos en el desarrollo?

Relacionado al algoritmo de `Canny`:
- La importancia de ajustar los parámetros de detección en función del brillo y el contraste del frame para mejorar la detección de bordes.

- La necesidad de considerar la diferencia entre frames consecutivos para evitar efectos de motion blur.

- La utilidad de la dilatación para mejorar la detección de bordes en presencia de ruido.

## Filtrado de bordes estáticos mediante análisis temporal

Durante las primeras pruebas observamos que el algoritmo de Canny detectaba gran cantidad de bordes pertenecientes al fondo (marcos, paredes, muebles, monitor, etc.).  
Estos bordes no aportan información relevante al seguimiento y generan *features* espurias que afectan negativamente la estabilidad del método de Lucas-Kanade.

Para resolver esto implementamos un **modelo temporal de contornos estáticos**, basado en:

- posición del contorno (centro del bounding box),
- tamaño aproximado (w, h),
- cantidad de apariciones acumuladas (`life`),
- última aparición (`last_seen`),
- tolerancia espacial y de tamaño (`STATIC_MATCH_DIST`, `STATIC_SIZE_TOL`).

La idea es sencilla:  
si un contorno aparece reiteradamente en la misma región del frame, se clasifica como **estático** y se descarta en las siguientes etapas del pipeline.

Este filtro se inspira en conceptos vistos en la materia sobre **persistencia temporal de gradientes espaciales**: si un borde pertenece a un objeto completamente inmóvil, su gradiente espacial se repite; mientras que los bordes de objetos móviles presentan variabilidad entre frames.

El resultado final es una máscara de contornos **dinámicos**, que excluye sistemáticamente el fondo y solo conserva estructura relevante del objeto a seguir.

---

## Manejo del motion blur mediante gradientes espacio–temporales

Cuando un objeto se mueve rápido, el desenfoque de movimiento reduce drásticamente los gradientes espaciales en las aristas perpendiculares a la dirección del movimiento.  
Esto hace que Canny pierda bordes importantes justamente en las zonas donde el seguimiento es más crítico.

Aunque la materia presenta técnicas de restauración (filtros inversos, Wiener, Richardson–Lucy), estas dependen de conocer o estimar la PSF asociada al blur de movimiento, y no son adecuadas para aplicar **en tiempo real** con una PSF distinta en cada frame.

En lugar de intentar reconstruir la imagen, adoptamos un método basado en **gradientes temporales**, consistente con la teoría del flujo óptico (uso conjunto de Ix, Iy, It).

## Selección y filtrado de features para Lucas-Kanade

Para la detección de puntos de interés utilizamos Shi–Tomasi, pero aplicando dos restricciones clave:

1. Solo se detectan points-of-interest dentro de regiones que contienen bordes dinámicos.
2. Filtrado de outliers en Lucas-Kanade eliminando desplazamientos no realistas.

## Visualización del sistema
El programa presenta cuatro vistas simultáneas:
- Frame original
- LK basado en movimiento global
- Bordes dinámicos filtrados
- LK basado exclusivamente en bordes

---

# Versión actual del proyecto (presentación final)

Esta sección describe la **versión actual del código**, correspondiente a la **presentación final** del trabajo práctico.  
La base conceptual se mantiene, pero se incorporaron mejoras sustanciales orientadas a robustez, estabilidad y capacidad de reconocimiento.

## Principales mejoras

### Inclusión de red neuronal (MobileNet-SSD)
Se incorporó una red neuronal convolucional (MobileNet-SSD) para el reconocimiento de objetos.

- La inferencia se ejecuta **únicamente cuando se detecta movimiento**.
- Además, se impone un **cooldown mínimo de 15 frames** desde la última detección para evitar inferencias redundantes y bajar el consumo de recursos de CPU.
- Las detecciones de la red neuronal se aceptan únicamente si su bounding box presenta intersección con regiones previamente identificadas como dinámicas, descartando detecciones asociadas a estructuras estáticas del fondo.

### Mejoras en el preprocesamiento orientado a Canny

En la primera versión del proyecto, los parámetros de Canny se ajustaban de forma **discreta**, dividiendo el rango de iluminación en intervalos.  
Para cada rango de brillo se utilizaba un conjunto fijo de parámetros, lo que introducía discontinuidades, requería calibración manual frecuente y resultaba poco robusto frente a variaciones suaves de iluminación.

En la versión actual, este enfoque fue reemplazado por una **función continua dependiente de la iluminación**.  
Se estima el **brillo global del frame** (valor medio de los píxeles) y, a partir de este, se calculan dinámicamente los parámetros de preprocesamiento utilizados por Canny.

La función adaptativa tiene como salidas:
- la **sensibilidad de Canny** (definición de umbrales bajo y alto a partir de la mediana),
- el **tamaño del kernel de blur gaussiano**, para atenuar ruido de compresión y ruido de sensor en baja iluminación,
- el **clip limit de CLAHE**, que controla la amplificación de contraste local,
- el **tamaño de los tiles de CLAHE**, que define la escala espacial de la ecualización.

De esta manera, los parámetros se ajustan **frame a frame de forma suave**, sin saltos abruptos, mejorando la estabilidad temporal de los bordes detectados y reduciendo la necesidad de calibración manual.  
Adicionalmente, se incorporó un **modo de calibración manual en tiempo real** mediante trackbars para análisis y validación experimental.


### Captura de video sin backlog
Se incorporó un **FrameGrabber en un hilo dedicado**, que mantiene siempre el último frame disponible, evitando acumulación de buffer cuando el procesamiento es más lento que la captura.

### Tracking de bounding boxes persistente
- Tracks con ID, tiempo de vida (TTL) y asociación por IoU.
- Actualización de bboxes por **unión** (no shrink).
- Merge de tracks por solapamiento o contención.

### Integración de flujo óptico en el movimiento de bboxes
Las bboxes pueden desplazarse utilizando el flujo óptico estimado sobre puntos en bordes, mejorando continuidad temporal.

### Mejoras de visualización y debug
- Overlay opcional de información de debug.
- Vista combinada 2x2 reescalada para presentación.

---
