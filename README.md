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

La matriz de probabilidades obtenida permite formar una secuencia de caracteres para comparar con los labels de las entradas. El problema más evidente es que el largo de la secuencia de salida obtenida es mayor que el largo de la secuencia target, dado que se tendrán tantos carácteres como time-frames tenga la secuencia de entrada. Además, se desconoce el alineamiento de la secuencia de salida con la de entrada.

El (algoritmo CTC)[https://distill.pub/2017/ctc/?undefined=&ref=assemblyai.com] computa la probabilidad de una secuencia de salida dada una entrada, y eso lo hace sumando las probabilidades de todos los *alineamientos permitidos*. 

Se introduce un caracter blank que permite tener instantes en donde no hay carácter y no colapsar caracteres que deben repetirse.

Luego, los *alineamientos permitidos* son todos aquellos que al reducir la secuencia de salida (al colapsar todos los caracteres repetidos que no esten separados por un blank), conducen a la secuencia target.

Una vez encontrados todos los *alineamientos permitidos*, se calcula la probabilidad de obtener la secuencia target como: $`P_{CTC}(C|X) = \sum_{Z\in \Lambda}\Pi_{t=1}^{T} P(z_t|X)`$,

donde `C` es la secuencia target, `X` es la secuencia de entrada, $`\Lambda`$ es el conjunto de alineamientos permitidos, y `T` es la cantidad de time-frames de la secuencia de entrada. Luego, la probabilidad $`P(z_t|X)`$ se puede obtener de la matriz de probabilidades obtenida a la salida de la red neuronal.

La función de Loss que se utilizó es: $`Loss(X, C)= -log(P_{CTC}(C|X))`$,

Y el criterio de minimización: $`Loss(W) = \frac{1}{N} \sum\limits_{i=1}^{N} Loss(X^i, C^i, W)`$, donde N es la cantidad de secuencias de entrada y W son los parámetros de la red.

### Optimizer

El optimizer que utilizó para actualizar los pesos es [AdamW](https://towardsdatascience.com/why-adamw-matters-736223f31b5d). 

Adam es una modificación de Stochastic Gradient Descent, que trackea a lo largo del tiempo (es decir, en cada mini-batch) la media movil del gradiente (llamado momento de primer orden $`m(t)`$) y de su cuadrado (llamado momento de segundo orden $`v(t)`$), los cuales dan una idea de cuánto esta variando el gradiente respecto de iteraciones anteriores. 

Salvando constantes multiplicativas, el paso con el cual se actualizan los parámetros del sistema es: $`w(t) = w(t-1) - \alpha\frac{m(t)}{\sqrt{v(t)}+\epsilon}`$

Recordando la definición de varianza: $`Var(x) = \langle x^2 \rangle - \langle x \rangle ^2`$. 

Si el gradiente no varía respecto de iteraciones anteriores, la varianza será cercana a 0 y $`\langle x^2 \rangle = v(t) \approx \langle x \rangle^2`$. Por lo tanto, se tendrá que $`\sqrt{v(t)} \approx m(t)`$. Luego, los parámetros se actualizarán según la constante $`\alpha`$. 

Al contrario, si el gradiente es muy variante y errático, la relación $`\frac{m(t)}{\sqrt{v(t)}}`$ será muy pequeña y la constante de acutalización sera mucho menor que $`\alpha`$.

En conclusión, cuando el gradiente varía mucho iteración a iteración, no es recomendable tomar pasos grandes, dado que la superficie es muy errática y si el paso es demasiado grande se puede cambiar los parámetros en una dirección equivocada. En cambio, cuando el gradiente es estable, se pueden tomar pasos más grandes ya que es más probable que se siga en un camino de descenso del gradiente.

Por otro lado, AdamW introdujo un parámetro de regularización llamado *weight decay*, que se basa en la idea de que las redes con pesos más pequeños tienden a tener menos *overfitting* y a generalizar mejor. Bajo esta premisa, se introduce un parámetro `\gamma` que define la importancia relativa entre minimizar la función de loss original (más importante si los pesos son pequeños) y encontrar pesos pequeños (más importante si los pesos son grandes). 

Luego se agrega un término $`\gamma w(t-1)`$ a la actualización de pesos y la actualización queda: $`w(t) = w(t-1) - \eta(\alpha\frac{m(t)}{\sqrt{v(t)}+\epsilon} + \gamma w(t-1))`$. Este *weight decay* reduce los pesos de manera exponencial, forzando a que la red aprenda pesos mas pequeños.

### Scheduler

El scheduler que se utilizó es el *One Cycle Learning Rete Scheduler*, que permite entrenar las redes de manera más rápida, manteniendo su capacidad de generalización. Consiste en definir una constante de aprendizaje que empiece siendo pequeña, luego aumentarla de manera lineal hasta un máximo, y luego reducirla nuevamente hasta el mínimo inicial de manera lineal.


## Inferencia

Esta red se debe poder utilizar con entradas cuyas etiquetas son desconocidas, con lo cual, no se puede utilizar el algoritmo de CTC para decodificar la salida. Para realizar la inferencia con entradas sin etiquetas, se utilizó un *Greedy Decoder*.

Este decoder consiste en encontrar una secuencia recorriendo el camino de mayor probabilidad marginal. Para cada tiempo, elige el caracter con mayor probabilidad. Luego se reduce la secuencia colapsando los caracteres repetidos y eliminando de la secuencia los caracteres *blank*. 

El Decoder se encuentra en el archivo [Decoder.py](/src/Decoder.py).

## Cómo entrenar un modelo

### Módulos necesarios
Los módulos de Python necesarios para correr el script son:

- Numpy
- Matplotlib
- Torch
- Torchaudio
- Jiwer

Se puede utilizar el archivo [requirements.txt](/src/requirements.txt) para instalar todos los módulos requeridos:

> PIP:
```
pip install -r requirements.txt
```

### Entrenamiento: endtoend_l40.py
El archivo [endtoend_l40.py](/src/endtoend_l40.py) realiza el entrenamiento de la red con el dataset `latino 40`. En caso de no utilizar este dataset, se tienen que reemplazar de dicho archivo, las lineas:

```python
    train_dataset = Latino40Dataset('../dataset/train.json', '../dataset')
    test_dataset = Latino40Dataset('../dataset/valid.json', '../dataset')

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn= lambda x: data_processing(x, 'train'),
                                **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)
```
Con el dataset deseado.

La arquitectura de la red se puede controlar mediante el diccionario `hparams` del mismo archivo:

```python
      hparams={
        "n_cnn_layers": 10,
        "n_rnn_layers": 3,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
```


Para correr el script hay que posicionarse en la carpeta de src y ejecutar el siguiente comando:
```
python -W ignore -u endtoend_l40.py | tee ../training_logs/log_BS_X_NCNN_Y_NRNN_Z.txt
```

donde `X` es el `batch_size`, `Y` es `n_cnn_layers` y `Z` es `n_rnn_layers`.

El script va a generar un log del training redireccionando el contenido de la stdoutput.

También se puede correr el script sin guardar el log:

```
python endtoend_l40.py
```

Además, el script guarda:
 - el modelo en la carpeta [model](/model), con la siguiente nomenclatura: `model_BS_X_CNNL_Y_NRNN_Z` donde `X` es el batch size, `Y` es la cantidad de capas convolucionales, y `Z` es la cantida de capas recurrentes.
 - un json con los vectores de loss de test, train, y CER (en ese orden), en la carpeta [training_logs](/training_logs), con la siguiente nomenclatura: `losses_log_BATCHSIZE_X_CNNL_X_RNNL_X.json`


## Performance

Para evaluar la performance del modelo se utilizó la medida CER: Character Error Rate.

Esta medida cuenta cuántas inserciones (I), deleciones (D) y sustituciones (S) se deben realizar sobre una secuencia para que sea igual a una secuencia target. Luego, se calcula el CER como: $`CER = \frac{S + D + I}{N}`$ donde N es el largo de la secuencia target.

Con el script [tesing.py](/src/model_assessment.py) se realiza inferencia sobre un test de validación/test utilizando un modelo guardado, y se imprimen en consola todas las secuencias predecidas junto con las etiquetas correspondientes. Además, se imprime la loss promedio y el CER promedio, y el número de parámetros totales de la red.

Para correr el script primero hay que modificar en el archivo [testing.py](/src/testing.py) los parámetros de `hparams` para cargar el modelo correspondiente. 
Luego, posicionarse en la carpeta src y ejecutar el siguiente comando:

```
python -W ignore -u testing.py | tee ../test_log/test_log_BS_X_NCNN_Y_NRNN_Z.txt
```

donde `X` es el `batch_size`, `Y` es `n_cnn_layers` y `Z` es `n_rnn_layers`.

El script va a generar un log del training redireccionando el contenido de la stdoutput.

También se puede correr el script sin guardar el log:

```
python testing.py
```



