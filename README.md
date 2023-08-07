# Automatic Speech-Recognition

Modelo end-to-end de reconocimiento automático del habla, basado en DeepSpeech II. 

El modelo tiene una arquitectura de capas de redes convolucionales seguidas de capas de redes recurrentes y feed forward. La salida del modelo es una matriz de probabilidades para cada caracter en cada instante de tiempo, y esas probabilidades se usan para decodificar la secuencia de texto más probable. 

El entrenamiento está basado en el algoritmo CTC.

### Arquitectura de la red

``

### Modules needed

- Numpy
- Matplotlib
- Torch
- Torchaudio
- Jiwer

### Running the script

En la carpeta de src:

`python -W ignore -u endtoend_l40.py | tee ../training_logs/log_BS_X_NCNN_Y_NRNN_Z.txt` 

donde `X` es el `BATCH SIZE`, `Y` es `NCNN LAYERS` y `Z` es `NRNN LAYERS`.

El script va a generar un log del training redireccionando el contenido de la stdoutput.

### Assessment of the model

El modelo se puede testear 
