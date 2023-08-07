# Automatic Speech-Recognition

Modelo end-to-end de reconocimiento automático del habla, basado en DeepSpeech II. 

El modelo tiene una arquitectura de capas de redes convolucionales seguidas de capas de redes recurrentes y feed forward. La salida del modelo es una matriz de probabilidades para cada caracter en cada instante de tiempo, y esas probabilidades se usan para decodificar la secuencia de texto más probable. 

El entrenamiento está basado en el algoritmo CTC.

## Arquitectura de la red

![Arquitectura de la red](/readme_pics/architecture.png)

## Pre-procesamiento

### Mel Spectrogram

Sobre las señales de audio crudas se aplica el Espectrograma Mel, que consiste en realizar el espectrograma de la señal mediante la FFT y luego transformar la escala de frecuencias a una escala logarítmica, de modo tal que se asemeje a la manera en que los humanos perciben las frecuencias de audio (mayor capacidad de dicernir fracuencias bajas que altas).

### Data Augmentation - SpecAugment

La tecnica utilizada consiste en cortar, en cada epoch y para cada sample, bloques aleatorios del espectrograma, tanto en el eje de las frecuencias como en el eje del tiempo. Esto genera una diversidad en el dataset lo cual aumenta su tamaño efectivo.

Se puede realizar en Pytorch con la función `FrecuencyMasking`

## Modules needed

- Numpy
- Matplotlib
- Torch
- Torchaudio
- Jiwer

## Running the script

En la carpeta de src:

`python -W ignore -u endtoend_l40.py | tee ../training_logs/log_BS_X_NCNN_Y_NRNN_Z.txt` 

donde `X` es el `BATCH SIZE`, `Y` es `NCNN LAYERS` y `Z` es `NRNN LAYERS`.

El script va a generar un log del training redireccionando el contenido de la stdoutput.

## Assessment of the model

El modelo se puede testear 
