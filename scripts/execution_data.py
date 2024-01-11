import json
import os
import datetime

myDictionary_loss = {}
myDictionary_acc = {}
loss_path = '/scratch/uduran005/tfg-workspace/graphics/loss_data.json'
acc_path = '/scratch/uduran005/tfg-workspace/graphics/acc_data.json'

def createAccLossData(dateTime, loss_values, accuracy_values, epoca):
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
    myDictionary_loss[key1_loss] = dateTime
    myDictionary_loss[key2_loss] = loss_values
    myDictionary_loss[key3_loss] = epoca
    general_list_loss.append(myDictionary_loss)

    json_path_acc = '/scratch/uduran005/tfg-workspace/graphics/'
    json_acc = 'acc_data.json'
    key1_acc = "date"
    key2_acc = "acc_values"
    key3_acc = "num_epochs"
    myDictionary_acc[key1_acc] = dateTime
    myDictionary_acc[key2_acc] = accuracy_values
    myDictionary_acc[key3_acc] = epoca
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
