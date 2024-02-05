import json
import os
import datetime

myDictionary_loss = {}
myDictionary_acc = {}
loss_path = '/scratch/uduran005/tfg-workspace/graphics/loss_data.json'
acc_path = '/scratch/uduran005/tfg-workspace/graphics/acc_data.json'

def createAccLossData(dateTime, loss_values, accuracy_values, epoca, hidden_dim, num_layers, num_heads, learning_rate, batch_size, weight_decay):
    # Se vacian los diccionarios
    myDictionary_acc.clear()
    myDictionary_loss.clear()

    # Se comprueba si existe el archivo previamente para crear su estructura
    if (not os.path.exists(loss_path) and not os.path.exists(acc_path)):
        general_list_loss = []
        general_list_acc = []
    else:
        general_list_loss = json.load(open(loss_path))
        general_list_acc = json.load(open(acc_path))


    # Se crean tanto el diccionario para accuracy como para loss
    json_path_loss = '/scratch/uduran005/tfg-workspace/graphics/'
    json_loss = 'loss_data.json'

    key1_loss = "date"
    key2_loss = "loss_values"
    key3_loss = "num_epochs"
    key4_loss = "hidden_dim"
    key5_loss = "num_layers"
    key6_loss = "num_heads"
    key7_loss = "learning_rate"
    key8_loss = "batch_size"
    key9_loss = "weight_decay"

    myDictionary_loss[key1_loss] = dateTime
    myDictionary_loss[key2_loss] = loss_values
    myDictionary_loss[key3_loss] = epoca
    myDictionary_loss[key4_loss] = hidden_dim
    myDictionary_loss[key5_loss] = num_layers
    myDictionary_loss[key6_loss] = num_heads
    myDictionary_loss[key7_loss] = learning_rate
    myDictionary_loss[key8_loss] = batch_size
    myDictionary_loss[key9_loss] = weight_decay

    general_list_loss.append(myDictionary_loss)

    json_path_acc = '/scratch/uduran005/tfg-workspace/graphics/'
    json_acc = 'acc_data.json'

    key1_acc = "date"
    key2_acc = "acc_values"
    key3_acc = "num_epochs"
    key4_acc = "hidden_dim"
    key5_acc = "num_layers"
    key6_acc = "num_heads"
    key7_acc = "learning_rate"
    key8_acc = "batch_size"
    key9_acc = "weight_decay"

    myDictionary_acc[key1_acc] = dateTime
    myDictionary_acc[key2_acc] = accuracy_values
    myDictionary_acc[key3_acc] = epoca
    myDictionary_acc[key4_acc] = hidden_dim
    myDictionary_acc[key5_acc] = num_layers
    myDictionary_acc[key6_acc] = num_heads
    myDictionary_acc[key7_acc] = learning_rate
    myDictionary_acc[key8_acc] = batch_size
    myDictionary_acc[key9_acc] = weight_decay

    general_list_acc.append(myDictionary_acc)

    # Se crea y rellena el archivo .json
    with open(os.path.join(json_path_loss, json_loss), 'w') as file_loss:
        json.dump(general_list_loss, file_loss)

    with open(os.path.join(json_path_acc, json_acc), 'w') as file_acc:
        json.dump(general_list_acc, file_acc)

# loss_values = [0.03, 0.02, 0.80, 0.005]
# acc_values = [0.4, 0.5, 1, 0.3]
# epoca = 4
# dateTime = datetime.datetime.now()
# dateTime = dateTime.strftime("%d%m%Y-%H%M")

# createAccLossData(dateTime, loss_values, acc_values, epoca)
