import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
from dataLoader import load_hdf5_data, createDataLoaders
import json
import os
from json_creator import createMapeo_Clases, createMapeo

## Se crea la clase Transformer
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))            ##Capa de embedding para el token CLS
        self.cls_token.requires_grad = True
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        cls_tokens = self.cls_token.expand(-1, x.size(1), -1) ## Expandir el token CLS para todas las muestras del batch
        #print(f"Entrada al embeding --> {x.size()}")
        x = self.embedding(x)
        x = torch.cat((cls_tokens, x), dim = 0) ## Agregar el token CLS al comienzo de la secuencia
        #print(f"Antes del permute --> {x.size()}")
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)
        #print(f"Después del permute --> {x.size()}")
        cls_output = x[0]               ## Suponiendo que el token CLS este al comienzo
        x = self.fc(cls_output)
        return F.log_softmax(x, dim=-1)


# Llamada a los metodos que se encargar de crear los DataLoaders customizados
file_path_train = 'C:/Universidad/TFG/Desarrollo/data_vector/TRAIN_landmarks_dataset.hdf5'  ##Path del archivos con las coordenadas recogidas (train)
file_path_val = 'C:/Universidad/TFG/Desarrollo/data_vector/VAL_landmarks_dataset.hdf5'      ##Path del archivos con las coordenadas recogidas (val)
file_path_test = 'C:/Universidad/TFG/Desarrollo/data_vector/TEST_landmarks_dataset.hdf5'    ##Path del archivos con las coordenadas recogidas (test)
path_mapeoClasses = 'C:/Universidad/TFG/Desarrollo/index/Mapeo_Clases.json'

# Se compueba si existe el archivo y si no es así, se genera
if (not os.path.exists(path_mapeoClasses)):
    createMapeo_Clases()

num_classes = len(json.load(open('C:/Universidad/TFG/Desarrollo/index/Mapeo_Clases.json')))

train_loader = createDataLoaders(num_classes, file_path_train)
val_loader =  createDataLoaders(num_classes, file_path_train)       ##Finalmente será file_path_val
test_loader = createDataLoaders(num_classes, file_path_train)       ##Finalmente será file_path_test


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


# Definir el modelo y otros hiperparámetros
input_dim = 225
hidden_dim = 224
num_layers = 2
num_heads = 4
output_dim = 2000       ##Coger el size del .json de mapeo (Mapeo_clases.json) 
dropout = 0.1
learning_rate = 0.001
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Evaluación del modelo
def evaluacion(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inp, tg, msk in loader:
            inp, tg, msk = inp.to(device), tg.to(device), msk.to(device)

            outputs = model(inp, msk)                       ##Modelo con el token CLS incluido
            print(f"Dimensiones --> {outputs.size()}")
            predicted = torch.argmax(outputs, 0)    ##Generar predicciones basadas en la salida del token CLS
            print(f"Predict --> {predicted.size()}\n\ntg --> {tg.size()}")
            total += tg.size(0)
            correct += (predicted == tg).sum().item()

    accuracy = correct / total
    print(f"[VALIDACION] - Accuracy on test set: {accuracy * 100:.2f}%")

# Entrenamiento del modelo
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inp, tg, msk in train_loader:
        inp, tg, msk = inp.to(device), tg.to(device), msk.to(device)

        optimizer.zero_grad()
        outputs = model(inp, msk)               ##Modelo con el token CLS incluido
        #print(f"Tamaño de la salida --> {outputs.size()}")
        #print(f"Tamaño tg --> {tg.size()}")
        loss = criterion(outputs, tg)        ##Usar la salida del token CLS para calcular la perdida
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    evaluacion(model, val_loader)
    print(f"[ENTRENAMIENTO] - Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
