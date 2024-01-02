import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
from dataLoader import load_hdf5_data, createDataLoaders
import json
import os
from json_creator import createMapeo_Clases, createMapeo
from test_draw_graph import drawLossGraphic, drawEvalGraphic
import datetime

## Se crea la clase Transformer
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, max_seq_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))            ##Capa de embedding para el token CLS
        self.cls_token.requires_grad = True
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_embedding = self.get_positional_embedding(hidden_dim, max_seq_len)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        cls_tokens = self.cls_token.expand(-1, x.size(1), -1) ## Expandir el token CLS para todas las muestras del batch
        #print(f"Entrada al embeding --> {x.size()}")
        x = self.embedding(x) + self.positional_embedding[:, :x.size(1)]  # Suma incrustaciones posicionales
        x = torch.cat((cls_tokens, x), dim = 0) ## Agregar el token CLS al comienzo de la secuencia
        #print(f"Antes del permute --> {x.size()}")
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)
        #print(f"Después del permute --> {x.size()}")
        cls_output = x[0]               ## Suponiendo que el token CLS este al comienzo
        x = self.fc(cls_output)
        return F.log_softmax(x, dim=-1)

    def get_positional_embedding(self, embed_size, max_seq_len):
        positional_embedding = torch.zeros(max_seq_len, embed_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term)
        return positional_embedding.unsqueeze(0)

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
learning_rate = 0.001
max_seq_len = 100
num_epochs = 999999999999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, output_dim, max_seq_len)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# EVALUACION
def evaluacion(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inp, tg, msk in loader:
            inp, tg, msk = inp.to(device), tg.to(device), msk.to(device)
            outputs = model(inp, msk)
            predicted = torch.argmax(outputs, dim=0)
            total += tg.size(1)
            correct += (predicted == tg.long()).sum().item()

    accuracy = correct / total if total != 0 else 0
    return accuracy

# ENTRENAMIENTO
epoch = 0
patience = 4
best_accuracy = 0.0
no_improvement_count = 0
loss_values = []
accuracy_values = []

while no_improvement_count < patience:
    model.train()
    running_loss = 0.0

    for inp, tg, msk in train_loader:
        inp, tg, msk = inp.to(device), tg.to(device), msk.to(device)

        optimizer.zero_grad()
        outputs = model(inp, msk)
        loss = criterion(outputs, tg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    loss_values.append(epoch_loss)
    accuracy = evaluacion(model, val_loader, device)
    accuracy_values.append(accuracy)
    epoca = epoch + 1
    print(f"[ENTRENAMIENTO] - Epoch [{epoca}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    epoch += 1

print(f"Se han mantenido el mismo resultado durante {patience + 1} epochs. Deteniendo el bucle.")
dateTime = datetime.datetime.now()
dateTime = dateTime.strftime("%d%m%Y")
drawLossGraphic(loss_values, epoca, dateTime)
drawEvalGraphic(accuracy_values, epoca, dateTime)