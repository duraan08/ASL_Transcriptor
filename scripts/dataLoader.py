import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py

# Dataset personalizado para cargar los datos
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Función para cargar datos desde un archivo HDF5
def load_hdf5_data(file_path):
    hdf5_file = h5py.File(file_path, 'r')
    for video in hdf5_file:
        print(f'Video encontrado --> {video}')
        train_data = torch.tensor(hdf5_file[video][:])

    return train_data

# # Ruta al archivo HDF5
# file_path = 'C:/Universidad/TFG/Desarrollo/data_vector/landmarks_dataset.hdf5'

# # Cargar datos desde el archivo HDF5
# train_data = load_hdf5_data(file_path)

# Funcion para crear los dataLoaders que se usaran en el transformer
def createDataLoaders(train_data):
    # Determinar el porcentaje para el conjunto de validación
    val_percentage = 0.2
    total_samples = train_data.size(0)
    print(f"Longitud datos de entrenamiento antes de separar --> {total_samples}")
    num_val_samples = int(total_samples * val_percentage)

    # Separar los datos en conjuntos de entrenamiento y validación
    val_data = train_data[:num_val_samples]
    train_data = train_data[num_val_samples:]

    # Dividir los datos en inputs y targets (asumiendo que están concatenados)
    train_inputs, train_targets = train_data[:, :-1], train_data[:, -1]
    val_inputs, val_targets = val_data[:, :-1], val_data[:, -1]

    # Crear datasets y dataloaders
    train_dataset = CustomDataset(train_inputs, train_targets)
    val_dataset = CustomDataset(val_inputs, val_targets)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # # Verificar la creación exitosa de los DataLoaders
    # for inputs, targets in train_loader:
    #     print(f"Batch de entrenamiento - Inputs: {inputs.shape}, Targets: {targets.shape}")
    #     print(f"\nPrimer lote de entrenamiento - Inputs: {inputs}")
    #     print(f"\nPrimer lote de entrenamiento - Targets: {targets}")
    #     break

    # for inputs, targets in val_loader:
    #     print(f"Batch de validacion - Inputs: {inputs.shape}, Targets: {targets.shape}")
    #     print(f"\nPrimer lote de validacion - Inputs: {inputs}")
    #     print(f"\nPrimer lote de validacion - Targets: {targets}")
    #     break

    return train_dataset, train_loader, val_dataset, val_loader
