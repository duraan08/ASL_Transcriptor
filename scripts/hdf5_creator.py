import os
import h5py
import numpy as np
from get_keypoints import main

# Ruta al video (TRAIN)
path_train = os.listdir('/scratch/uduran005/tfg-workspace/ASL_videos/train/')
path_train.sort()
str_path_train = '/scratch/uduran005/tfg-workspace/ASL_videos/train/'

# Ruta al video (TEST)
path_test = os.listdir('/scratch/uduran005/tfg-workspace/ASL_videos/test/')
path_test.sort()
str_path_test = '/scratch/uduran005/tfg-workspace/ASL_videos/test/'

# Ruta al video (VAL)
path_val = os.listdir('/scratch/uduran005/tfg-workspace/ASL_videos/val/')
path_val.sort()
str_path_val = '/scratch/uduran005/tfg-workspace/ASL_videos/val/'

## Path para pequeña prueba
path_lil = os.listdir('/scratch/uduran005/tfg-workspace/ASL_videos/lil_test/')  ##lil_test
path_lil.sort()
str_path_lil = '/scratch/uduran005/tfg-workspace/ASL_videos/t/'  ##lil_test

## Path para almacenar keypoints
path_npy = ''

#paths = [str_path_train, str_path_test, str_path_val, str_path_lil]
paths = [str_path_test] ##str_path_lil
for selected_path in paths:
    path_splited = selected_path.split('/')
    if (path_splited[5] == "train"):
        hd5f_file = h5py.File('/scratch/uduran005/tfg-workspace/data_vector/TRAIN_landmarks_dataset.hdf5', 'w')
        path_npy = '/scratch/uduran005/tfg-workspace/keypoints/TRAIN'
    elif (path_splited[5] == "test"):
        hd5f_file = h5py.File('/scratch/uduran005/tfg-workspace/data_vector/TEST_landmarks_dataset.hdf5', 'w')
        path_npy = '/scratch/uduran005/tfg-workspace/keypoints/TEST'
    elif (path_splited[5] == "val"):
        hd5f_file = h5py.File('/scratch/uduran005/tfg-workspace/data_vector/VAL_landmarks_dataset.hdf5', 'w')
        path_npy = '/scratch/uduran005/tfg-workspace/keypoints/VAL'
    else:
        hd5f_file = h5py.File('/scratch/uduran005/tfg-workspace/data_vector/LIL_landmarks_dataset.hdf5', 'w')       ##Antes era LIL_landmarks_dataset.hdf5 
        path_npy = '/scratch/uduran005/tfg-workspace/keypoints/LIL'   ##Antes era LIL
    
    path = os.listdir(selected_path)
    str_path = selected_path

## Se generan los diferentes archivos .npy por video con sus keypoints correspondientes
if (not bool(os.listdir(path_npy))):
    main(str_path, path_npy)
else:
    print("Elimina los elementos del directorio")

## Se crea el archivo hdf5 sobre el cual se va a escribir
path = os.listdir(path_npy) # Se obtiene la lista de los keypoints recogidos en su fichero .npy

## Por cada video registrado con su matriz de keypoints se almacena en el archivo hdf5
for selected_video in path:
    data = np.load(f'{path_npy}/{selected_video}')
    hd5f_file.create_dataset(f"{selected_video}", data=data)
hd5f_file.close()
