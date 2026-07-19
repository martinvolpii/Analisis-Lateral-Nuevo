# Análisis Lateral de Marcha Murina con DeepLabCut

Pipeline modular en Python para analizar marcha murina en vista lateral a partir de coordenadas generadas por DeepLabCut.

El flujo está diseñado para procesar videos de locomoción en cinta de correr, detectar ciclos de marcha, calcular goniometría lateral, estimar variables temporales, generar controles visuales y producir un Excel final de validación estadística.

---
> **Configuración actual del proyecto P30:** `LIKELIHOOD_MIN = 0.70`, `FPS = 60`, `SMOOTH_WINDOW = 11`. El script 04 reconoce automáticamente IDs como `856_P30`, `857_P30`, etc., evitando agruparlos como `UNKNOWN`.



## Objetivo del pipeline

Este repositorio permite analizar coordenadas de DeepLabCut correspondientes a los puntos anatómicos:

- `crest`
- `hip`
- `knee`
- `ankle`
- `foot`
- `toe`

A partir de estos puntos, el pipeline calcula:

- Coordenadas limpias.
- Eventos y ciclos de marcha.
- Perfiles de ciclo normalizados a 0–100%.
- Ángulos articulares laterales.
- Rangos angulares por ciclo.
- Tiempo de zancada.
- Porcentaje de apoyo.
- Porcentaje de oscilación.
- Toe clearance.
- Estadística descriptiva por ciclo y por animal.
- Normalidad.
- Dataset balanceado a 10 ciclos por animal.

---

## Estructura recomendada del repositorio

```text
Analisis-Lateral-Nuevo/
│
├── 01_preprocesamiento_y_ciclos.py
├── 02_goniometria_lateral_por_ciclos.py
├── 03_variables_temporales_y_toe_clearance.py
├── 04_validacion_estadistica_y_excel.py
├── README.md
```

---

## Requisitos

Usar Python 3.9 o superior.

Instalar dependencias:

```bash
pip install pandas numpy scipy matplotlib openpyxl xlsxwriter tables
```

Dependencias principales:

```text
pandas
numpy
scipy
matplotlib
openpyxl
xlsxwriter
tables
```

La librería `tables` es necesaria para leer archivos `.h5` de DeepLabCut.

---

## Formatos de entrada

El pipeline acepta archivos de DeepLabCut en formato:

```text
.h5
.csv
```

Se recomienda usar preferentemente `.h5`, porque conserva mejor la estructura original de DeepLabCut:

```text
scorer → bodyparts → coords
```

Cada punto anatómico debe tener:

```text
x
y
likelihood
```

---

## Configuración oficial recomendada para videos a 60 fps

Para el experimento oficial se recomienda grabar y analizar todos los videos a:

```text
60 fps
```

Esto mejora la resolución temporal de la marcha y ayuda a capturar mejor eventos rápidos como foot strike, toe-off y swing.

Si los videos fueron grabados a 60 fps, los scripts temporales deben ejecutarse con:

```bash
--fps 60
```

También se recomienda dejar estos valores por defecto dentro de los scripts:

### En `01_preprocesamiento_y_ciclos.py`

```python
FPS = 60.0
LIKELIHOOD_MIN = 0.70
MAX_GAP_INTERPOLATION = 10
SMOOTH_WINDOW = 11
MERGE_TOLERANCE_FRAMES = 4
```

### En `03_variables_temporales_y_toe_clearance.py`

```python
FPS = 60.0
SUSTAIN_FRAMES = 4
```

Los parámetros definidos en segundos no necesitan duplicarse:

```python
MIN_CYCLE_DURATION_S = 0.12
MAX_CYCLE_DURATION_S = 1.50
MIN_STANCE_DURATION_S = 0.05
MIN_SWING_DURATION_S = 0.05
```

Estos valores se convierten internamente a frames usando el FPS indicado.

---

## Orden general de ejecución

El análisis debe ejecutarse en este orden:

```text
01 → preprocesamiento, limpieza y ciclos
02 → goniometría lateral por ciclo
03 → variables temporales y toe clearance
04 → Excel de validación estadística, normalidad y datos balanceados
```

Flujo completo:

```text
Archivo DeepLabCut .h5/.csv
        ↓
01_preprocesamiento_y_ciclos.py
        ↓
*_clean_coords.csv
*_gait_cycles.csv
        ↓
02_goniometria_lateral_por_ciclos.py
        ↓
*_cycle_angle_ranges.csv
*_cycle_angle_profiles.csv
        ↓
03_variables_temporales_y_toe_clearance.py
        ↓
*_gait_temporal_by_cycle.csv
        ↓
04_validacion_estadistica_y_excel.py
        ↓
Excel final de validación estadística
```

---

# 1. Preprocesamiento y detección de ciclos

Archivo:

```text
01_preprocesamiento_y_ciclos.py
```

Este script realiza:

- Lectura de archivos `.h5` o `.csv` de DeepLabCut.
- Extracción de puntos anatómicos.
- Filtro por `likelihood`.
- Interpolación de gaps cortos.
- Suavizado de coordenadas.
- Detección de eventos de marcha.
- Construcción de ciclos de marcha.
- Normalización de ciclos a 0–100%.
- Exportación de archivos de control.

## Definición del ciclo

La detección usa principalmente los puntos distales:

```text
toe
foot
```

El método recomendado es:

```text
distal_x
```

El ciclo se define como el intervalo entre dos eventos consecutivos del mismo tipo en la trayectoria distal.

En la versión corregida para este montaje experimental, el inicio del ciclo se fija usando:

```python
EVENT_POLARITY = "min"
```

Esto evita que el modo `auto` elija el extremo opuesto y desfase el 0% del ciclo. Si cambia el lado corporal, la cámara o la orientación del montaje, se debe validar nuevamente con el PNG de control.

## Uso básico a 60 fps

```bash
python 01_preprocesamiento_y_ciclos.py "archivo_DLC.h5" --fps 60 --outdir salida_01_ciclos
```

También puede usarse con CSV:

```bash
python 01_preprocesamiento_y_ciclos.py "archivo_DLC.csv" --fps 60 --outdir salida_01_ciclos
```

## Parámetros importantes

```text
--fps
    Frames por segundo del video. Para el experimento oficial usar 60.

--likelihood-min
    Umbral mínimo de confianza para aceptar coordenadas. Default oficial: 0.70.

--max-gap
    Longitud máxima configurada para interpolación de gaps cortos. Default: 10 frames.

--smooth-window
    Ventana de suavizado; se fuerza a impar. Default oficial: 11.

--event-method
    Método de detección de eventos. Recomendado: distal_x.

--event-polarity
    Polaridad del evento. Para este montaje: min.

--prominence
    Sensibilidad para detectar eventos.

--contact-bodyparts
    Puntos distales usados para construir la señal de contacto. Recomendado: toe,foot.

--rhythm-bodyparts
    Puntos auxiliares exportados para control del ritmo; no definen el contacto.

--outdir
    Carpeta donde se guardarán los resultados.
```

## Archivos de salida

```text
*_clean_coords.csv
*_events_detected.csv
*_gait_cycles.csv
*_normalized_cycles.csv
*_cycle_detection_signals.csv
*_cycle_detection_check.png
*_params.txt
```

## Archivo más importante para revisar

```text
*_cycle_detection_check.png
```

Este gráfico debe revisarse visualmente antes de continuar. Si los ciclos no están correctamente detectados, no se debe avanzar al análisis angular ni temporal sin corregir primero el problema.

---

# 2. Goniometría lateral por ciclos

Archivo:

```text
02_goniometria_lateral_por_ciclos.py
```

Este script usa las salidas del script 01 y calcula:

- Ángulos frame a frame.
- Perfiles angulares por ciclo.
- Perfiles normalizados a 0–100% del ciclo.
- Rangos angulares por ciclo.
- Resumen angular por video.
- Gráficos de perfiles y control.

Este script no detecta ciclos nuevos. Usa exclusivamente los ciclos generados por el script 01.

## Definición de ángulos

Los ángulos se calculan usando tres puntos anatómicos:

```text
hip angle:
    crest - hip - knee

knee angle:
    hip - knee - ankle

ankle angle:
    knee - ankle - foot

foot angle:
    ankle - foot - toe
```

Las unidades son grados:

```text
°
```

## Uso básico

```bash
python 02_goniometria_lateral_por_ciclos.py \
  "salida_01_ciclos/archivo_clean_coords.csv" \
  --cycles "salida_01_ciclos/archivo_gait_cycles.csv" \
  --outdir salida_02_angulos
```

El script 02 no necesita argumento `--fps`, porque calcula ángulos por frame y por ciclo. Los tiempos ya dependen del script 01.

## Archivos de salida

```text
*_frame_angles.csv
*_cycle_angle_profiles.csv
*_cycle_angle_ranges.csv
*_angle_range_summary.csv
*_angle_profiles.png
*_angle_cycle_control.png
*_goniometry_params.txt
```

## Archivo más importante para revisar

```text
*_angle_cycle_control.png
```

Este gráfico permite verificar si los ciclos detectados cortan correctamente las señales angulares.

---

# 3. Variables temporales y toe clearance

Archivo:

```text
03_variables_temporales_y_toe_clearance.py
```

Este script usa las salidas del script 01 y calcula:

- Tiempo de zancada.
- Duración de apoyo.
- Duración de oscilación.
- Porcentaje de apoyo.
- Porcentaje de oscilación.
- Toe clearance.

Este script no detecta ciclos nuevos. Usa los ciclos generados por el script 01.

## Uso básico a 60 fps

```bash
python 03_variables_temporales_y_toe_clearance.py \
  "salida_01_ciclos/archivo_clean_coords.csv" \
  --cycles "salida_01_ciclos/archivo_gait_cycles.csv" \
  --fps 60 \
  --outdir salida_03_temporal
```

## Variables calculadas

```text
stride_duration_s
    Tiempo de zancada en segundos.

stance_duration_s
    Duración de apoyo en segundos.

swing_duration_s
    Duración de oscilación en segundos.

stance_percent
    Porcentaje del ciclo en fase de apoyo.

swing_percent
    Porcentaje del ciclo en fase de oscilación.

toe_clearance_px
    Elevación máxima del toe durante oscilación, en píxeles.
```

## Unidades

```text
Tiempos:
    segundos (s)

Porcentajes:
    % del ciclo de marcha

Toe clearance:
    píxeles (px)
```

Para convertir `toe_clearance_px` a milímetros se necesita una calibración espacial del video:

```text
mm por píxel
```

o

```text
píxeles por mm
```

## Archivos de salida

```text
*_gait_temporal_by_cycle.csv
*_gait_temporal_video_summary.csv
*_gait_temporal_control.png
*_toe_clearance_control.png
*_temporal_params.txt
```

## Archivos más importantes para revisar

```text
*_gait_temporal_control.png
*_toe_clearance_control.png
```

Estos gráficos permiten revisar si la separación entre apoyo y oscilación es coherente y si el cálculo de toe clearance es razonable.

---

# 4. Validación estadística y Excel general

Archivo:

```text
04_validacion_estadistica_y_excel.py
```

Este script no vuelve a procesar coordenadas ni recalcula ciclos. Usa las salidas de los scripts 02 y 03:

```text
*_cycle_angle_ranges.csv
*_gait_temporal_by_cycle.csv
```

Objetivo:

- Unificar resultados por ciclo.
- Conservar todos los animales, datasets y ciclos encontrados.
- Crear tablas por animal.
- Calcular medias por animal.
- Calcular descriptivos generales.
- Evaluar normalidad.
- Generar un Excel ordenado y auditable.

## Política para los datos reales P30

El script 04 **no excluye animales**, **no excluye datasets** y **no deduplica automáticamente**.
Las banderas de control de calidad (`accepted_temporal`, `reject_reason`, etc.) se conservan
en el Excel, pero las filas originales permanecen disponibles para auditoría.

## Uso básico

```bash
python 04_validacion_estadistica_y_excel.py \
  --input-dir carpeta_resultados_pipeline \
  --out validacion_estadistica_dlc.xlsx
```

El script reconoce IDs históricos tipo `R1`, `R2` y el formato actual
`856_P30...`, `857_P30...`; para este último usa `856`, `857`, etc. como
`animal_id` y conserva el nombre completo como `dataset_id`.

## Unidad estadística

La unidad estadística principal debe ser:

```text
animal
```

No el ciclo individual.

Los ciclos individuales se conservan para revisar dispersión intra-animal, pero la inferencia grupal debe realizarse usando medias por animal.

## Conservación de datos

La hoja `cycles_individual_all` contiene todos los ciclos descubiertos por el pipeline 02/03.
No se recorta a un número fijo de ciclos por animal en el script 04.

La hoja `data_retention_all` sirve como auditoría explícita de que cada animal y dataset
permanece incluido.

## Hojas principales del Excel

```text
README
dataset_qc_all
data_retention_all
cycles_individual_all
cycles_long_finite
stats_by_animal
animal_means
general_stats_n_animal
normality_cycles_desc
```

La hoja más importante para análisis posterior es:

```text
animal_means
```

---

## Variables principales del análisis

### Variables angulares

```text
hip_range_deg
knee_range_deg
ankle_range_deg
foot_range_deg
```

Unidad:

```text
grados (°)
```

### Variables temporales

```text
stride_duration_s
stance_duration_s
swing_duration_s
stance_percent
swing_percent
```

Unidades:

```text
segundos (s)
porcentaje (%)
```

### Variable espacial

```text
toe_clearance_px
```

Unidad:

```text
píxeles (px)
```

---

## Control de calidad obligatorio

Antes de aceptar un video para análisis, revisar:

```text
1. Video etiquetado de DeepLabCut.
2. Likelihood de toe, foot, ankle, knee, hip y crest.
3. *_cycle_detection_check.png
4. *_angle_cycle_control.png
5. *_gait_temporal_control.png
6. *_toe_clearance_control.png
```

Un video debe considerarse dudoso si presenta:

```text
- pérdida frecuente del toe o foot
- ciclos mal cortados
- eventos de contacto desplazados
- pocas zancadas válidas
- toe clearance incoherente
- fases de apoyo/oscilación visualmente incorrectas
- grandes saltos de coordenadas
```

---

## Recomendaciones para grabación oficial

Para maximizar el rendimiento de DeepLabCut:

```text
- Usar 60 fps en todos los videos.
- Mantener cámara completamente lateral.
- Mantener misma distancia y altura de cámara.
- Usar misma resolución en todos los videos.
- Evitar sombras sobre las patas.
- Evitar reflejos en la cinta.
- Mantener buena iluminación.
- Mantener buen contraste entre animal, pata y fondo.
- Registrar velocidad de cinta.
- Registrar grupo, animal, fecha y condición experimental.
- No mezclar videos de 30 fps y 60 fps dentro del mismo análisis temporal.
```

---

## Consideraciones metodológicas

Para análisis oficial:

```text
- Usar el mismo modelo DeepLabCut final para todos los animales.
- Usar el mismo snapshot para todos los videos oficiales.
- No ajustar parámetros mirando un grupo específico.
- No excluir animales o datasets automáticamente; cualquier decisión futura debe ser explícita, documentada y realizada fuera del script 04.
- Usar media por animal como unidad estadística.
- Usar ciclos individuales solo para descriptivos y visualización.
```

Para comparaciones futuras, por ejemplo WT vs SOD1, se recomienda que el Excel final tenga una columna:

```text
grupo
```

Ejemplo:

```text
animal_id | grupo | variable | media_animal
WT01      | WT    | hip_range_deg
SOD101    | SOD1  | hip_range_deg
```

La comparación estadística debe realizarse entre animales, no entre ciclos.

---

## Flujo recomendado para experimento oficial

```text
1. Grabar todos los videos a 60 fps.
2. Analizar todos los videos con el mismo modelo DeepLabCut.
3. Ejecutar script 01 con --fps 60.
4. Revisar PNG de detección de ciclos.
5. Ejecutar script 02.
6. Revisar control angular.
7. Ejecutar script 03 con --fps 60.
8. Revisar control temporal y toe clearance.
9. Ejecutar script 04 para Excel de validación y normalidad conservando todos los datos.
10. Usar medias por animal para análisis estadístico final.
```

---

## Ejemplo de ejecución completa para un video

```bash
python 01_preprocesamiento_y_ciclos.py \
  "videos/WT01_DLC_filtered.h5" \
  --fps 60 \
  --outdir resultados/01_WT01

python 02_goniometria_lateral_por_ciclos.py \
  "resultados/01_WT01/WT01_DLC_filtered_clean_coords.csv" \
  --cycles "resultados/01_WT01/WT01_DLC_filtered_gait_cycles.csv" \
  --outdir resultados/02_WT01

python 03_variables_temporales_y_toe_clearance.py \
  "resultados/01_WT01/WT01_DLC_filtered_clean_coords.csv" \
  --cycles "resultados/01_WT01/WT01_DLC_filtered_gait_cycles.csv" \
  --fps 60 \
  --outdir resultados/03_WT01
```

Después de procesar todos los animales:

```bash
python 04_validacion_estadistica_y_excel.py \
  --input-dir resultados \
  --out validacion_estadistica_dlc.xlsx
```

---

## Nota final

Este pipeline está pensado para que el análisis sea reproducible, auditable y consistente entre animales. La decisión más importante para evitar sesgos es mantener constantes el modelo DeepLabCut, los parámetros de análisis y las reglas de control de calidad durante todo el experimento oficial. El script 04 de datos reales conserva todos los animales, datasets y ciclos.
