
# Análisis Lateral Nuevo

Pipeline modular en Python para el análisis cinemático lateral de marcha en ratones a partir de archivos generados por DeepLabCut.

Este repositorio está diseñado para procesar coordenadas de puntos anatómicos obtenidas desde videos laterales de locomoción en cinta de correr. El flujo de trabajo está dividido en scripts cortos e independientes para reducir errores, facilitar la revisión visual y permitir correcciones específicas en cada etapa del análisis.

## Objetivo del pipeline

El objetivo principal es analizar la marcha murina utilizando coordenadas de DeepLabCut, específicamente los puntos anatómicos:

- `crest`
- `hip`
- `knee`
- `ankle`
- `foot`
- `toe`

A partir de estos puntos, el pipeline permite:

- Preprocesar coordenadas.
- Detectar ciclos de marcha.
- Calcular ángulos articulares.
- Calcular rangos angulares por ciclo.
- Calcular variables temporales de marcha.
- Calcular toe clearance.
- Exportar archivos de control y revisión.

## Estructura del repositorio

```text
Analisis-Lateral-Nuevo/
│
├── 01_preprocesamiento_y_ciclos.py
├── 02_goniometria_lateral_por_ciclos.py
├── 03_variables_temporales_y_toe_clearance.py
├── README.md
Requisitos

Este pipeline requiere Python 3.9 o superior.

Instalar dependencias:

pip install pandas numpy scipy matplotlib tables

Dependencias principales:

pandas
numpy
scipy
matplotlib
tables

La librería tables es necesaria para leer archivos .h5 de DeepLabCut.

Formatos de entrada

El pipeline acepta archivos de salida de DeepLabCut en formato:

.h5
.csv

Se recomienda trabajar preferentemente con archivos .h5, ya que conservan mejor la estructura original de DeepLabCut:

scorer → bodyparts → coords

Cada punto anatómico debe tener coordenadas:

x
y
likelihood
Orden general de ejecución

El análisis debe ejecutarse en este orden:

1. 01_preprocesamiento_y_ciclos.py
2. 02_goniometria_lateral_por_ciclos.py
3. 03_variables_temporales_y_toe_clearance.py

El primer script genera las coordenadas limpias y los ciclos de marcha.
El segundo script usa esos ciclos para calcular goniometría.
El tercer script usa los mismos ciclos para calcular variables temporales y toe clearance.

1. Preprocesamiento y detección de ciclos

Archivo:

01_preprocesamiento_y_ciclos.py

Este script realiza:

- Lectura de archivos .h5 o .csv de DeepLabCut.
- Extracción de puntos anatómicos.
- Filtro por likelihood.
- Interpolación de coordenadas.
- Suavizado de coordenadas.
- Detección de ciclos de marcha.
- Normalización de ciclos a 0–100%.
- Exportación de archivos de control.
Método de detección de ciclos

La detección de ciclos se basa principalmente en los puntos distales:

toe
foot

El script construye una señal distal combinada usando la trayectoria horizontal del pie en el eje x.

La lógica utilizada es:

Un ciclo de marcha = intervalo entre dos eventos consecutivos del mismo pie.

Los puntos hip, knee y ankle no se usan como contacto directo, pero se exportan como señales auxiliares para revisar el patrón locomotor.

Uso básico
python 01_preprocesamiento_y_ciclos.py "archivo_DLC.h5" --fps 30 --outdir salida_01_ciclos

También puede usarse con CSV:

python 01_preprocesamiento_y_ciclos.py "archivo_DLC.csv" --fps 30 --outdir salida_01_ciclos
Parámetros importantes
--fps
    Frames por segundo del video. Por defecto se recomienda usar 30.

--likelihood-min
    Umbral mínimo de confianza para aceptar coordenadas.

--cycle-bodypart
    Punto principal usado para revisar ciclos.

--event-polarity
    Permite elegir si los eventos se detectan como máximos o mínimos de la señal.

--outdir
    Carpeta donde se guardarán los resultados.
Archivos de salida
*_clean_coords.csv
*_events_detected.csv
*_gait_cycles.csv
*_normalized_cycles.csv
*_cycle_detection_signals.csv
*_cycle_detection_check.png
*_params.txt
Archivo más importante para revisar
*_cycle_detection_check.png

Este gráfico debe revisarse visualmente antes de pasar al segundo y tercer script.
Si los ciclos no están bien detectados, no se debe continuar con el análisis angular ni temporal.

2. Goniometría lateral por ciclos

Archivo:

02_goniometria_lateral_por_ciclos.py

Este script realiza:

- Lectura de coordenadas limpias.
- Lectura de ciclos detectados.
- Cálculo de ángulos frame a frame.
- Segmentación angular por ciclo.
- Normalización angular a 0–100% del ciclo.
- Cálculo de rango angular por ciclo.
- Cálculo de media, desviación estándar y SEM por video.
- Exportación de tablas y gráficos.

Este script no detecta ciclos nuevos. Usa exclusivamente los ciclos generados por el script 01.

Definición de ángulos

Los ángulos se calculan usando tres puntos anatómicos:

hip angle:
    crest - hip - knee

knee angle:
    hip - knee - ankle

ankle angle:
    knee - ankle - foot

foot angle:
    ankle - foot - toe
Rango angular

El rango angular se calcula por ciclo como:

rango angular = ángulo máximo - ángulo mínimo

Se calcula para:

hip
knee
ankle
foot
Uso básico
python 02_goniometria_lateral_por_ciclos.py "salida_01_ciclos/archivo_clean_coords.csv" --outdir salida_02_angulos

En caso de querer indicar manualmente el archivo de ciclos:

python 02_goniometria_lateral_por_ciclos.py "salida_01_ciclos/archivo_clean_coords.csv" \
  --cycles "salida_01_ciclos/archivo_gait_cycles.csv" \
  --outdir salida_02_angulos
Archivos de salida
*_frame_angles.csv
*_cycle_angle_profiles.csv
*_cycle_angle_ranges.csv
*_angle_range_summary.csv
*_angle_profiles.png
*_angle_cycle_control.png
*_goniometry_params.txt
Descripción de salidas principales
Ángulos frame a frame
*_frame_angles.csv

Contiene los ángulos articulares calculados en cada frame.

Perfiles angulares por ciclo
*_cycle_angle_profiles.csv

Contiene los perfiles angulares normalizados de cada ciclo desde 0 hasta 100%.

Rangos angulares por ciclo
*_cycle_angle_ranges.csv

Contiene el ángulo mínimo, máximo y rango angular de cada articulación en cada ciclo.

Resumen angular por video
*_angle_range_summary.csv

Contiene el promedio, desviación estándar y error estándar de la media de los rangos angulares por video.

Variables incluidas:

hip_range_mean_deg
hip_range_sd_deg
hip_range_sem_deg

knee_range_mean_deg
knee_range_sd_deg
knee_range_sem_deg

ankle_range_mean_deg
ankle_range_sd_deg
ankle_range_sem_deg

foot_range_mean_deg
foot_range_sd_deg
foot_range_sem_deg
Archivo más importante para revisar
*_angle_cycle_control.png

Este gráfico permite verificar si los ciclos detectados cortan correctamente las señales angulares.

3. Variables temporales y toe clearance

Archivo:

03_variables_temporales_y_toe_clearance.py

Este script calcula únicamente:

- Tiempo de zancada.
- Porcentaje de apoyo.
- Porcentaje de oscilación.
- Toe clearance.

Este script no detecta ciclos nuevos. Usa los ciclos generados por el script 01.

Variables calculadas
Tiempo de zancada
Tiempo entre dos contactos consecutivos del mismo pie.

Se expresa en segundos.

Porcentaje de apoyo
Porcentaje del ciclo en que la extremidad se encuentra en fase de apoyo.
Porcentaje de oscilación
Porcentaje del ciclo en que la extremidad se encuentra en fase de oscilación.

Se calcula como:

swing_percent = 100 - stance_percent
Toe clearance
Elevación máxima del punto toe durante la fase de oscilación.

Se expresa inicialmente en pixeles.

Uso básico
python 03_variables_temporales_y_toe_clearance.py "salida_01_ciclos/archivo_clean_coords.csv" --fps 30 --outdir salida_03_temporal

En caso de querer indicar manualmente el archivo de ciclos:

python 03_variables_temporales_y_toe_clearance.py "salida_01_ciclos/archivo_clean_coords.csv" \
  --cycles "salida_01_ciclos/archivo_gait_cycles.csv" \
  --fps 30 \
  --outdir salida_03_temporal
Archivos de salida
*_gait_temporal_by_cycle.csv
*_gait_temporal_video_summary.csv
*_gait_temporal_control.png
*_toe_clearance_control.png
*_temporal_params.txt
Descripción de salidas principales
Variables por ciclo
*_gait_temporal_by_cycle.csv

Contiene una fila por ciclo con:

cycle_id
start_frame
end_frame
stride_duration_s
stance_duration_s
swing_duration_s
stance_percent
swing_percent
toe_clearance_px
Resumen por video
*_gait_temporal_video_summary.csv

Contiene resumen del video con media, desviación estándar y SEM:

stride_duration_s_mean
stride_duration_s_sd
stride_duration_s_sem

stance_percent_mean
stance_percent_sd
stance_percent_sem

swing_percent_mean
swing_percent_sd
swing_percent_sem

toe_clearance_px_mean
toe_clearance_px_sd
toe_clearance_px_sem
Archivos más importantes para revisar
*_gait_temporal_control.png
*_toe_clearance_control.png

Estos gráficos permiten revisar si la separación entre apoyo y oscilación es coherente y si el cálculo de toe clearance es correcto.

Flujo completo de análisis

Ejemplo de ejecución completa para un archivo .h5:

python 01_preprocesamiento_y_ciclos.py "archivo_DLC.h5" --fps 30 --outdir salida_01_ciclos

Luego:

python 02_goniometria_lateral_por_ciclos.py "salida_01_ciclos/archivo_clean_coords.csv" --outdir salida_02_angulos

Luego:

python 03_variables_temporales_y_toe_clearance.py "salida_01_ciclos/archivo_clean_coords.csv" --fps 30 --outdir salida_03_temporal
Recomendaciones importantes
1. Revisar siempre los gráficos de control

Antes de interpretar los resultados, revisar:

*_cycle_detection_check.png
*_angle_cycle_control.png
*_gait_temporal_control.png
*_toe_clearance_control.png

Si los ciclos están mal detectados, los resultados posteriores no deben usarse.

2. No tratar ciclos como animales independientes

Los ciclos de marcha son repeticiones dentro del mismo video.
Para análisis estadístico grupal, se debe usar el promedio por video, animal o estadio, no cada ciclo como una observación independiente.

3. Mantener los scripts separados

Cada script cumple una función específica:

01 = ciclos
02 = ángulos
03 = variables temporales

Esto permite corregir errores de forma localizada sin modificar todo el pipeline.

4. No subir datos pesados al repositorio

Se recomienda no subir archivos grandes como:

.h5
.csv
.mp4
.mov
.avi
.png

Los datos originales y resultados deben almacenarse localmente o en una carpeta externa.

Archivos sugeridos para ignorar en Git

Se recomienda usar un archivo .gitignore con:

*.h5
*.mp4
*.avi
*.mov
*.csv
*.png
salida_*/
resultados/
__pycache__/
.ipynb_checkpoints/
