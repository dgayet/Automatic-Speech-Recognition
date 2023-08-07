# Automatic Speech-Recognition

Modelo end-to-end de reconocimiento automático del habla, basado en DeepSpeech II. 

El modelo tiene una arquitectura de capas de redes convolucionales seguidas de capas de redes recurrentes y feed forward. La salida del modelo es una matriz de probabilidades para cada caracter en cada instante de tiempo, y esas probabilidades se usan para decodificar la secuencia de texto más probable. 

El entrenamiento está basado en el algoritmo CTC.

## Arquitectura de la red

![Arquitectura de la red](/readme_pics/architecture.png)

## Pre-procesamiento

Todo el pre-procesamiento aplicado al dataset se encuentra en el módulo [preprocessing.py](/src/preprocessing.py)

### Mel Spectrogram

Sobre las señales de audio crudas se aplica el Espectrograma Mel, que consiste en realizar el espectrograma de la señal mediante la FFT y luego transformar la escala de frecuencias a una escala logarítmica, de modo tal que se asemeje a la manera en que los humanos perciben las frecuencias de audio (mayor capacidad de dicernir fracuencias bajas que altas).

Se utilizaron 128 coeficientes.

En Pytorch se aplica con la función `MelSpectrogram`

### Data Augmentation - SpecAugment

La tecnica utilizada consiste en cortar, en cada epoch y para cada sample, bloques aleatorios del espectrograma, tanto en el eje de las frecuencias como en el eje del tiempo. Esto genera una diversidad en el dataset lo cual aumenta su tamaño efectivo.

Se puede realizar en Pytorch con la función `FrecuencyMasking`

### Label Mapping

A cada etiqueta de cada señal de audio se la transformó a una secuencia de números enteros para trabajar directamente con números en vez de caracteres. 
En las etiquetas se reemplazó el caracter `ñ` por `ni`, y además se agregó un caracter asociado al espacio `<SPACE>`.

Para realizar estó se creó una clase `TextTransforms` que se encuentra en el archivo [Classes.py](/src/Classes.py)

## Modelo 
El modelo de toda la arquitectura se encuentra en el archivo [Model.py](/src/Model.py).

- La clase `ResidualCNN` corresponde a las capas de redes convolucionales
- La clase `Bidirectional GRU` correspodne a las capas de redes recurrentes
- La clase `SpeechRecognitionModel` es la arquitectura completa, que comprende las redes convolucionales, recurrentes, y las redes feed-forward intermedias y finales.

### N Residual Convolutional Layer

Se utilizó un Kernel de 3x3 con 32 canales de salida con un stride unitario. La cantidad de capas es un parámetro a eleción.

Como es una arquitectura de red residual, se agregó una conexión directa en paralelo desde la entrada hacia la salida de la red.

La función de activación es una [GELU](https://medium.com/@shauryagoel/gelu-gaussian-error-linear-unit-4ec59fb2e47c): es la función de probabilidad acumulada de una normal estándar y es diferenciable en todo su dominio (al contrario de la RELU que no es diferenciable en 0).

Las técnicas de regularización utilizadas son:

- Dropout: consiste en apagar cada neurona con una probabilidad *p* durante el entrenamiento. Esto permite que las neuronas no sean tan dependientes de sus neronas vecinas, reduciendo así el *overfitting*.
- [Layer Normalization](https://www.pinecone.io/learn/batch-layer-normalization/): estandariza las entradas calculando la media y el desvio estándar sobre **la dimensión de los features**. Es más aplicable que batch normalization cuando el tamaño del batch size es pequeño. Se aplica sobre el mapa de activación, antes de la función de activación.

A la salida de la última capa de redes convolucionales hay una red feed-forward que se utiliza para acondicionar la salida de la red convolucional para que sea adecuada como entrada de la red recurrente.

### N Bi-direccional GRU Layers

Se utilizó una red recurrente del tipo GRU, es decir, con compuertas de _update_ y _reset_ que dictan cuánto del estado oculto en un timepo anterior y cuánto de la entrada en el tiempo actual se utiliza para generar el estado oculto en el tiempo actual. 

La red es de tipo bi-direccional, es decir, la salida es la concatenación de la salida de dos redes: una que procesa la información con la secuencia dese t=0 hasta t=T, y otra que procesa la secuencia inversa.

Se utilizó un estado oculto de tamaño 512. La canitdad de capas es un parámetro a elección.

También se utilizaron Layer Normalization y Dropout como métodos de regularización.

### N Linear Layer

A la salida de la Red Recurrente se introdujo dos capas de redes feed-forward que hacen las veces de clasificador. La función de activación utilizada en la última capa es una `softmax`, que transforma la salida en una mapa de probabilidades. 

## Entrenamiento

El entrenamiento se encuentra en la función `train_step.py` del archivo [engine.py](/src/engine.py).

### Criterio de minimización: CTC Loss

### Optimizer

### Scheduler

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
