import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
import json
import os
from json_creator import createMapeo

# Dataset personalizado para cargar los datos
class CustomDataset(Dataset):
    def __init__(self, num_classes, file_path):
        self.inputs, self.targets, self.mask = load_hdf5_data(file_path)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        target = torch.tensor(self.targets[idx])
        one_hot_target = F.one_hot(target, num_classes=self.num_classes)

        return self.inputs[idx], one_hot_target, self.mask[idx]

# Función para cargar datos desde un archivo HDF5
def load_hdf5_data(file_path):
    hdf5_file = h5py.File(file_path, 'r')
    
    train_data = []
    train_targets = []
    train_mask = []
    
    path_fichero = 'C:/Universidad/TFG/Desarrollo/index/Mapeo.json'
    if (not os.path.exists(path_fichero)):
        createMapeo()
    
    fichero = open(path_fichero) ## Mapeo clases
    fichero_json = json.load(fichero)

    ## Ficheros .json (Mapeo de cada gloss --> book = 0, ... || Acceso rapido --> 00000.mp4 = 0 [book])
    ## Aqui solo se necesita el segundo archivo

    for video in hdf5_file:
        print(f'Video encontrado --> {video}')
        train_data.append(torch.tensor(hdf5_file[video][:], dtype=torch.float))
        train_mask.append(torch.zeros(train_data[-1].size(0) + 1))
        train_targets.append(int(fichero_json[video])) ##La clase que sea [0, 2, 5] 0: book, 2: car, ...

    #print(f"Prueba dataLoader --> {train_targets}")
    hdf5_file.close()
    return train_data, train_targets, train_mask

# Funcion para crear los dataLoaders que se usaran en el transformer
def createDataLoaders(num_classes, file_path):
    # Crear datasets y dataloaders
    dataset = CustomDataset(num_classes, file_path)

    batch_size = 32

    def collate_fn_padd(batch):
        data = [x[0] for x in batch]
        data = pad_sequence(data, batch_first=False, padding_value=-2.0)
        #print(f"Datos --> {data.size()}")

        targets = [x[1] for x in batch]
        targets = torch.stack(targets).float()  # Convertir la lista de tensores en un solo tensor
        #print(f"Targets --> {targets.size()}")

        mask = [x[2] for x in batch]
        mask = pad_sequence(mask, batch_first=False, padding_value=1)
        #print(f"Mascara --> {mask.size()}")

        return data, targets, mask

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn_padd)

    return loader