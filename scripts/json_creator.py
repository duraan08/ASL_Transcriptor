import json
import os

data_path = json.load(open('/scratch/uduran005/tfg-workspace/index/WLASL.json'))
myDictionary = {}

def createMapeo_Clases():
    myDictionary.clear()
    json_path = '/scratch/uduran005/tfg-workspace/index'
    json_file_name = 'Mapeo_Clases.json'

    count = 0
    for i in data_path:
        key = count                 ## Se determina la clave
        val = i['gloss']            ## Se determina el valor
        myDictionary[key] = val     ## Se crea el diccionario
        count += 1                  ## Se aumenta el valor de la clave

    ##print(myDictionary)

    with open(os.path.join(json_path, json_file_name), 'w') as file:
        json.dump(myDictionary, file)

    # json_created_file = json.load(open('C:/Universidad/TFG/Desarrollo/index/Mapeo_Clases.json'))
    # print(json_created_file['11'])

def createMapeo():
    json_path = '/scratch/uduran005/tfg-workspace/index'
    json_file_name = 'Mapeo.json'
    myDictionary.clear()
    
    count = 0
    for i in data_path:
        val = count
        instancias = i['instances']
        for j in instancias:
            key = j['video_id'] + '.npy'        ##Antes era .mp4
            myDictionary[key] = val
        count += 1

    with open(os.path.join(json_path, json_file_name), 'w') as file:
        json.dump(myDictionary, file)

    # json_created_file = json.load(open('C:/Universidad/TFG/Desarrollo/index/Mapeo.json'))
    # print(f"{json_created_file['book'][0]}")
