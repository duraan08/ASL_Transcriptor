import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import h5py
import os
import numpy as np

# Obtener las coordenadas
def get_landmarks(landmarks, landmark_type):
        landmark_data = []
        if landmarks:
            for landmark in landmarks.landmark:
                print(f"{landmark_type} - X: {landmark.x}, Y: {landmark.y}, Z: {landmark.z}")
                landmark_data.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmark_data)

## Se crea el archivo hd5f para almacenar los landmarks
hd5f_file = h5py.File('C:/Universidad/TFG/Desarrollo/data_vector/landmarks_dataset.hdf5', 'w')

# Se establece la función encargada de obtener y dibujar los resultados
def detectAll(holistic, video_path, draw=False, display=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video")
        return
    
    ## Se alamcenan las coordenadas de la mano derecha, mano izquierda, pose
    v = video_path.split('/')
    print(f"\n\nVideo {v[len(v) - 1]} \n")

    ## Array dummy para cuando se de el caso de que las manos no aparezcan 
    hands_dummy = np.zeros([21,3])
    hands_dummy.fill(-2)

    pose_dummy = np.zeros([33,3])
    pose_dummy.fill(-2)

    print(f"size pose dummy --> {pose_dummy.shape} \n size manos dummy --> {hands_dummy.shape}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_copy = frame.copy()
        image_in_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultant = holistic.process(image_in_RGB)

        ##Obtener los landmarks de la mano derecha, izquierda y pose 
        right_hand_landmarks_raw = get_landmarks(resultant.right_hand_landmarks, "Right Hand")
        left_hand_landmarks_raw = get_landmarks(resultant.left_hand_landmarks, "Left Hand")
        pose_landmarks_raw = get_landmarks(resultant.pose_landmarks, "Pose")

        ## Comprobar que se hayan recogido las coordenadas y en caso de no ser así se da un valor ficticio
        if (len(right_hand_landmarks_raw) == 0):
            right_hand_landmarks_raw = hands_dummy
        if (len(left_hand_landmarks_raw) == 0):
            left_hand_landmarks_raw = hands_dummy
        if (len(pose_landmarks_raw) == 0):
            pose_landmarks_raw = pose_dummy

        print(f"Size pose --> {pose_landmarks_raw.shape} \n Size mano derecha --> {right_hand_landmarks_raw.shape} \n mano izquierda --> {left_hand_landmarks_raw.shape}")

        ##Aplanar pose_landmarks y avitar las 3 dimensiones
        pose_landmarks = [x for list in pose_landmarks_raw for x in list]
        right_hand_landmarks = [x for list in right_hand_landmarks_raw for x in list]
        left_hand_landmarks = [x for list in left_hand_landmarks_raw for x in list]

        pose = np.asarray(pose_landmarks)
        md = np.asarray(right_hand_landmarks)
        mi = np.asarray(left_hand_landmarks)
        print(f"Size pose --> {pose.shape} \n Size mano derecha --> {md.shape} \n mano izquierda --> {mi.shape}")

        ##Generar matriz [x,y] y agregar los landmarks
        frame_landmarks = right_hand_landmarks + left_hand_landmarks + pose_landmarks
        matrix.append(frame_landmarks)

        if draw:
            mp.solutions.drawing_utils.draw_landmarks(img_copy, resultant.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(img_copy, resultant.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(img_copy, resultant.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

        if display:
            ##cv2.imshow("Output", img_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            return img_copy, resultant
    

    cap.release()
    cv2.destroyAllWindows()
    return matrix

# Ruta al video
path = os.listdir('C:/Universidad/TFG/Desarrollo/ASL_videos/train/')
str_path = 'C:/Universidad/TFG/Desarrollo/ASL_videos/train/'

## Path para pequeña prueba
path_lil_test = os.listdir('C:/Universidad/TFG/Desarrollo/ASL_videos/lil_test/')
path_lil_test.sort()
str_path_lil_test = 'C:/Universidad/TFG/Desarrollo/ASL_videos/lil_test/'

##Se crea un array con los videos para poder acceder a ellos de manera ordenada
video_list_ordinal = []
## Se inicializa la matriz para almacenar los landmarks
matrix = []

for selected_video in path_lil_test:    
    ## Se inserta el nombre de cada video
    video_list_ordinal.append(selected_video)

    ## Se llama a la función para procesar el video
    matrix_v2 = detectAll(mp.solutions.holistic.Holistic(static_image_mode=False, min_tracking_confidence=0.7, min_detection_confidence=0.7),
          str_path_lil_test + selected_video, draw=True, display=True)
    
    ##Almacenar los landmarks obtenidos en el archivo hd5f
    hd5f_file.create_dataset(f"{selected_video}", data=np.asarray(matrix_v2))
## Se cierra el archivo al finalizar
hd5f_file.close()


# for elem in matrix:
#     print(len(elem[0]), len(elem[1]), len(elem[2]))


# detectAll(mp.solutions.holistic.Holistic(static_image_mode=False, min_tracking_confidence=0.7, min_detection_confidence=0.7),
#           path_prueba, draw=True, display=True)
