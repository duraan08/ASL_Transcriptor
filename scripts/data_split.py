import json
import os
import shutil

## Se obtiene el fichero que contiene los metadatos de los videos como el id, url ...
fichero = open('C:/Universidad/TFG/Desarrollo/index/WLASL.json')
## Se obtiene el directorio que almacena todos los videos 
lista_videos = os.listdir('C:/Universidad/TFG/Desarrollo/ASL_videos/all_videos')

## Se carga en su formato JSON
fichero_json = json.load(fichero)
##Se obtiene el id y su etiqueta split que indica si ese vídeo sirve para train, test, val
objeto = fichero_json[0]['instances'][0]['video_id'] + ".mp4"

## listas en las que se almacenaran los video segun su etiqueta split
index_video = []
## Se obtiene la ifnormación ('train', 'test', 'val') y su video id
count = 0
for i in fichero_json:
    instancias = i['instances']
    for j in instancias:
        index_video.append([j['video_id'] + ".mp4", j['split']])

print(f"Numero de videos --> {len(index_video)}")

## Se mueven los videos a su carpeta correspondiente
for v in index_video:
    for video in lista_videos:
        if (v[0] == video):
            if (v[1] == "train"):
                shutil.copy('C:/Universidad/TFG/Desarrollo/ASL_videos/' + video, 
                            'C:/Universidad/TFG/Desarrollo/ASL_videos/train/' + video)
            elif (v[1] == "test"):
                shutil.copy('C:/Universidad/TFG/Desarrollo/ASL_videos/' + video, 
                            'C:/Universidad/TFG/Desarrollo/ASL_videos/test/' + video)
            elif (v[1] == "val"):
                shutil.copy('C:/Universidad/TFG/Desarrollo/ASL_videos/' + video, 
                            'C:/Universidad/TFG/Desarrollo/ASL_videos/val/' + video)
            else:
                print(f"Existe otro tipo --> {v[1]}")