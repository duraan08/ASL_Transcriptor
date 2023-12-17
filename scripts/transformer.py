import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
from dataLoader import load_hdf5_data, createDataLoaders

## Se crea la clase Transformer
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout = 0.1):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    ## Metodo de propagación
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim = 1)
        x = self.dropout(x)
        x = self.fc(x)

        return F.log_softmax(x, dim = -1)

# Llamada a los metodos que se encargar de crear los DataLoaders customizados
file_path = 'C:/Universidad/TFG/Desarrollo/data_vector/landmarks_dataset.hdf5' # Path del archivos con las coordenadas recogidas
train_data = load_hdf5_data(file_path)
train_dataset, train_loader, val_dataset, val_loader =  createDataLoaders(train_data)


# Verificar la creación exitosa de los DataLoaders
for inputs, targets in train_loader:
    print(f"Batch de entrenamiento - Inputs: {inputs.shape}, Targets: {targets.shape}")
    print(f"\nPrimer lote de entrenamiento - Inputs: {inputs}")
    print(f"\nPrimer lote de entrenamiento - Targets: {targets}")
    break

for inputs, targets in val_loader:
    print(f"Batch de validacion - Inputs: {inputs.shape}, Targets: {targets.shape}")
    print(f"\nPrimer lote de validacion - Inputs: {inputs}")
    print(f"\nPrimer lote de validacion - Targets: {targets}")
    break


# Definir el modelo y otros hiperparámetros
input_dim = 50  # Reemplaza con la dimensión real de tus datos de entrada
hidden_dim = 64
num_layers = 2
num_heads = 4
output_dim = 5  # Reemplaza con el número real de clases de salida
dropout = 0.1
learning_rate = 0.001
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Entrenamiento del modelo
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * inputs.size(0)

#     epoch_loss = running_loss / len(train_loader.dataset)
#     print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

# # Evaluación del modelo
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, targets in test_loader:
#         inputs, targets = inputs.to(device), targets.to(device)

#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += (predicted == targets).sum().item()

# accuracy = correct / total
# print(f"Accuracy on test set: {accuracy * 100:.2f}%")

