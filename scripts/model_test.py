import torch 
import h5py
import datetime
import datasets
import os
import numpy as np
import json

def test(model, loader, device, criterion, hdf5_file):
    ## Para posteriormente utilizarlo y poder indentificar los diferentes archivos
    dateTime = datetime.datetime.now()
    dateTime = dateTime.strftime("%d%m%Y")

    ## Obtener Mapeo_Clases.json para conocer el valor de la respuesta
    path_json = '/scratch/uduran005/tfg-workspace/index/Mapeo_Clases.json'
    with open(path_json) as json_file:
        mapeo_clases = json.load(json_file)

    ## Comienza la evaluación
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
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

            # Calculando top-5 accuracy
            _, pred = outputs.topk(5, 1, True, True) # Obtiene los 5 índices más altos de las predicciones
            # Asegurarse de que tg tenga la forma correcta antes de expandir
            tg = tg.unsqueeze(1).expand_as(pred)
            correct += pred.eq(tg).sum().item() # Compara si los índices de las predicciones están en las etiquetas objetivo
            total += tg.size(0) # Suma el número de etiquetas objetivo

    epoch_loss = running_loss / len(loader.dataset)
    accuracy_test = metric.compute(predictions=predicciones, references=referencias)
    accuracy_test = accuracy_test['accuracy'] * 100

    # Calcula la precisión top-5
    top5_accuracy = correct / total * 100

    ## Obtener los nombres de los videos de cada hdf5
    with h5py.File(hdf5_file, "r") as f:
        video_ids = list(f.keys())
    
    ## Escribir un archivo con los resultados
    path_fichero = f"/scratch/uduran005/tfg-workspace/resultados/output_glosas_{dateTime}_1.txt"
    path_general = f"/scratch/uduran005/tfg-workspace/resultados"
    if (not os.path.exists(path_fichero)):
        with open(f'/scratch/uduran005/tfg-workspace/resultados/output_glosas_{dateTime}_1.txt', 'w') as output_file:
            for idx, prediction in enumerate(predicciones):
                video_id = video_ids[idx] # Esta lista debe ser proporcionada o generada previamente
                output_line = f"Video ID: {video_id}, Predicción : {prediction}, Glosas: {mapeo_clases[str(prediction.cpu().item())]}\n"
                output_file.write(output_line)
    else:
        file_list = os.listdir(path_general)
        for file in file_list:
            if (file.split('_')[2] == dateTime):
                file_index = file.split('_')[3].split('.')[0]
        
        with open(f'/scratch/uduran005/tfg-workspace/resultados/output_glosas_{dateTime}_{int(file_index)+1}.txt', 'w') as output_file:
            for idx, prediction in enumerate(predicciones):
                video_id = video_ids[idx] # Esta lista debe ser proporcionada o generada previamente
                output_line = f"Video ID: {video_id}, Predicción : {prediction}, Glosas: {mapeo_clases[str(prediction.cpu().item())]}\n"
                output_file.write(output_line)

    return accuracy_test, epoch_loss, top5_accuracy
