# Automatic Speech-Recognition

## Running the script

En la carpeta de src:

`python -W ignore -u endtoend_l40.py | tee ../training_logs/log_BS_X_NRNN_Y.txt` 

donde `X` es el `BATCH SIZE` e `Y` es `NRNN LAYERS` 

## Modules needed

- Numpy
- Matplotlib
- Torch
- Torchaudio
- Jiwer
