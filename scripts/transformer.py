import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import json
import os
import datetime
import datasets
import sys
from torch.utils.data import DataLoader, Dataset
from json_creator import createMapeo_Clases
from dataLoader import createDataLoaders
from draw_graph import drawGraph
from model_test import test

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## Se crea la clase Transformer
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, max_seq_len, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))            ##Capa de embedding para el token CLS
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.cls_token.requires_grad = True
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_embedding = self.get_positional_embedding(hidden_dim, max_seq_len).to(device)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward, dropout = dropout, batch_first = True)     ##batch_first = True --> Va primero
        #torch.nn.xavier_uniform_(encoder_layers)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        ## Incializar de forma uniforme los pesos  
        for name, param in self.transformer_encoder.named_parameters():
            ##print(f"Name : {name} // Param: {param}")
            if 'weight' in name and param.data.dim() == 2:
                torch.nn.init.xavier_uniform_(param, gain = torch.nn.init.calculate_gain("relu"))       ##Deberia de ser sqrt(2)

        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.dropout = nn.Dropout(dropout)
        torch.nn.init.kaiming_normal_(self.fc.weight, nonlinearity = "relu") 
        
        ## Otar forma de cls token
        #self.cls_emb = nn.Embedding(1, self.d_hidden)
        #torch.nn.init.normal_(self.cls_emb.weight, std = 0.02)

    def forward(self, x):     #Antes ,mask
        x = x.to(device)
        #print(f"Input shape --> {x.shape}")
        
        cls_token = self.cls_token.expand(-1, x.size(1), -1) ## Expandir el token CLS para todas las muestras del batch
        #print(f"Entrada al embeding --> {x.size()}")
        x = torch.cat((cls_token, x), dim = 0) ## Agregar el token CLS al comienzo de la secuencia
        #print(f"Shape de la secuencia despues de añadir el token CLS: {x.shape}")
        
        x = self.embedding(x) + self.positional_embedding[:, :x.size(1)]  # Suma incrustaciones posicionales
        x = x.permute(1, 0, 2)
        #print(f"Antes del encoder --> {x.size()}")
        #x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        x = self.transformer_encoder(x)
        #print(f"Shape de la salida del autoatento después de pasar la secuencia: {x.shape}")
        
        x = x.permute(1, 0, 2)
        #print(f"Después del encoder --> {x.size()}")
        
        cls_output = x[0]               ## Suponiendo que el token CLS este al comienzo
        #print(f"Shape de la salida del token CLS: {cls_output.shape}")
        
        x = self.fc(cls_output)
        return F.log_softmax(x, dim=-1)

    def get_positional_embedding(self, embed_size, max_seq_len):
        positional_embedding = torch.zeros(max_seq_len, embed_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term[:embed_size // 2])

        return positional_embedding.unsqueeze(0)

# Se establece que los datos se moveran a la GPU en caso de que este disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir hiperparametros
input_dim = 225
hidden_dim = 225
dim_feedforward = hidden_dim * 4
batch_size = 100
num_layers = 8
num_heads = hidden_dim // 64
weight_decay = 0.0001               ##Antes era 0
transformer_dropout = 0.1   ##Antes era 0
learning_rate = 0.0001  
output_dim = 2000       ##Coger el size del .json de mapeo (Mapeo_clases.json) 
max_seq_len = 100
num_epochs = 999999999999


# Llamada a los metodos que se encargar de crear los DataLoaders customizados
file_path_train = '/scratch/uduran005/tfg-workspace/data_vector/TRAIN_landmarks_dataset.hdf5'  ##Path del archivos con las coordenadas recogidas (train)
file_path_val = '/scratch/uduran005/tfg-workspace/data_vector/VAL_landmarks_dataset.hdf5'      ##Path del archivos con las coordenadas recogidas (val)
file_path_test = '/scratch/uduran005/tfg-workspace/data_vector/TEST_landmarks_dataset.hdf5'    ##Path del archivos con las coordenadas recogidas (test)
file_path_lil_test = '/scratch/uduran005/tfg-workspace/data_vector/t_landmarks_dataset.hdf5' ##Path del archivos con las coordenadas recogidas (lil test)
path_mapeoClasses = '/scratch/uduran005/tfg-workspace/index/Mapeo_Clases.json'

# Se compueba si existe el archivo y si no es así, se genera
if (not os.path.exists(path_mapeoClasses)):
    createMapeo_Clases()

num_classes = len(json.load(open('/scratch/uduran005/tfg-workspace/index/Mapeo_Clases.json')))

print(f"DataLoader con los datos del TRAIN: ")
train_loader = createDataLoaders(num_classes, file_path_train, device, batch_size)      ##Finalmente será file_path_train
print(f"DataLoader con los datos del EVALUACION: ")
val_loader =  createDataLoaders(num_classes, file_path_val, device, batch_size)       ##Finalmente será file_path_val

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


# Definir el modelo
model = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, output_dim, max_seq_len, dim_feedforward, dropout = transformer_dropout)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

## Para posteriormente utilizarlo y poder indentificar los diferentes archivos
dateTime = datetime.datetime.now()
dateTime = dateTime.strftime("%d%m%Y")

# EVALUACION
def evaluacion(model, loader, device, criterion):
    model.eval()
    running_loss =  0.0
    correct =  0
    total =  0
    metric = datasets.load_metric('accuracy')
    predicciones = []
    referencias = []
    loss = 0

    with torch.no_grad():
        for inp, tg in loader:
            inp, tg = inp.to(device), tg.to(device)
            referencias.extend(torch.argmax(tg, dim=1))
            outputs = model(inp)
            tg = torch.argmax(tg, dim=1)
            predicciones.extend(torch.argmax(outputs, dim=1))
            loss = criterion(outputs, tg)
            running_loss += loss.item()

    epoch_loss = running_loss / len(loader.dataset)
    accuracy_eval = metric.compute(predictions=predicciones, references=referencias)
    accuracy_eval = accuracy_eval['accuracy'] *  100

    return accuracy_eval, epoch_loss

# ENTRENAMIENTO
epoch = 0
patience = 9
best_accuracy = 0.0
no_improvement_count = 0
loss_values = []
accuracy_values_train = []
accuracy_values_test = []
loss_values_test = []
metric = datasets.load_metric('accuracy')

while no_improvement_count < patience:  ##Comprobación para EARLY_STOPPING

    predicciones = []
    referencias = []    ##Valor real 
    model.train()
    running_loss = 0.0

    for inp, tg in train_loader:       # Antes ,msk
        #inp, tg, msk = inp.to(device), tg.to(device), msk.to(device)
        inp, tg = inp.to(device), tg.to(device)
        #print(f"Device is --> {device}")
        optimizer.zero_grad()
        referencias.extend(torch.argmax(tg, dim = 1))
        #outputs = model(inp, msk)
        outputs = model(inp)
        tg = torch.argmax(tg, dim = 1)
        #outputs = torch.argmax(outputs, dim = 1)
        predicciones.extend(torch.argmax(outputs, dim = 1))
        loss = criterion(outputs, tg)
        
        # # Habilitar deteccion de anomalias
        # with torch.autograd.set_detect_anomaly(True):
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_values.append(epoch_loss)
    accuracy_eval, loss_eval = evaluacion(model, val_loader, device, criterion)
    predicciones = [x.item() for x in predicciones]         ##Se pasa a entero
    referencias = [x.item() for x in referencias]           ##Se pasa a entero

    accuracy_train = metric.compute(predictions = predicciones, references = referencias)
    accuracy_train = accuracy_train['accuracy'] * 100
    accuracy_values_train.append(accuracy_train)
    accuracy_values_test.append(accuracy_eval)
    loss_values_test.append(loss_eval)
    
    epoca = epoch + 1

    print(f"[ENTRENAMIENTO] - Epoch [{epoca}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {accuracy_train:.4f}")       ## Multiplicarlo x100
    print(f"[EVALUACION]  - Loss: {loss_eval:.4f} - Accuracy: {accuracy_eval:.4f}")

    if accuracy_train > best_accuracy:
        ## Guardar el modelo
        PATH = "/scratch/uduran005/tfg-workspace/model/modelo.pth"
        torch.save(model.state_dict(), PATH)

            # state = {
    #         'epoch': epoch,
    #         'state_dict': self.model.state_dict(),
    #         'optimiser': self.optimiser.state_dict(),
    #     }
 
    #     output_dir = self.args.root_dir + self.args.output_dir + self.args.saved_models_dir + self.args.last_model_dir 
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     torch.save(
    #         state, 
    #         output_dir + self.args.model_state_file
    #     )

        best_accuracy = accuracy_train
        no_improvement_count = 0

        ##Almacenar hyperparametros//mejor acc en train//mejor acc en eval//Resultado de test
        with open(f'/scratch/uduran005/tfg-workspace/model/datos_mejor_modelo_{dateTime}.txt', 'w') as file:
            file.write(f"[HIPERPARAMETROS]")
            file.write(f"\n - hidden_dim = {hidden_dim}")
            file.write(f"\n - num_layers = {num_layers}")
            file.write(f"\n - num_heads = {num_heads}")
            file.write(f"\n - learning_rate = {learning_rate}")
            file.write(f"\n - batch_size = {batch_size}")
            file.write(f"\n - weight_decay = {weight_decay}")
            file.write(f"\n - dropout = {transformer_dropout}")
            file.write(f"\n\n[ENTRENAMIENTO] - Epoch [{epoca}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {accuracy_train:.4f}")
            file.write(f"\n[EVALUACION]  - Loss: {loss_eval:.4f} - Accuracy: {accuracy_eval:.4f}")

    else:
        no_improvement_count += 1

    epoch += 1

print(f"Se han mantenido el mismo resultado durante {patience + 1} epochs. Deteniendo el bucle.\n\n")

## Se carga el loader de TEST
print(f"DataLoader con los datos del TEST: ")
test_loader = createDataLoaders(num_classes, file_path_test, device, batch_size)       ##Finalmente será file_path_test

#se comienza con la evaluación
#evaluacion_modelo(model, val_loader, device)
PATH_modelo = "/scratch/uduran005/tfg-workspace/model/modelo.pth"
model.load_state_dict(torch.load(PATH))
accuracy_test_model, loss_test_model = test(model, test_loader, device, criterion, file_path_test)
print(f"[EVALUACION]  - Loss: {loss_test_model:.4f} - Accuracy: {accuracy_test_model:.4f}")

with open(f'/scratch/uduran005/tfg-workspace/model/datos_mejor_modelo.txt', 'a') as file:
    file.write(f"\n[TEST]  - Loss: {loss_test_model:.4f} - Accuracy: {accuracy_test_model:.4f}")

##Dibujar graficas y almacenarlas como .pdf
drawGraph(loss_values, loss_values_test, epoca, dateTime, hidden_dim, num_layers, num_heads, learning_rate, batch_size, weight_decay, transformer_dropout, "loss")
drawGraph(accuracy_values_train, accuracy_values_test, epoca, dateTime, hidden_dim, num_layers, num_heads, learning_rate, batch_size, weight_decay, transformer_dropout, "acc")
