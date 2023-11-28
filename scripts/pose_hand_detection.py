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
hd5f_file = h5py.File('C:/Universidad/TFG/Desarrollo/data_vector/landmarks_data.hdf5', 'w')

# Se establece la función encargada de obtener y dibujar los resultados
def detectAll(holistic, video_path, draw=False, display=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video")
        return
    
    ## Se alamcenan las coordenadas de la mano derecha, mano izquierda, pose
    v = video_path.split('/')
    print(f"\n\nVideo {v[len(v) - 1]} \n")
    
    count = 0 ##Para diferenciar dentro del mismo grupo (Mano izquierda, derecha y pose) por cada frame acpturado
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_copy = frame.copy()
        image_in_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultant = holistic.process(image_in_RGB)

        ##Obtener los landmarks de la mano derecha, izquierda y pose 
        right_hand_landmarks = get_landmarks(resultant.right_hand_landmarks, "Right Hand")
        left_hand_landmarks = get_landmarks(resultant.left_hand_landmarks, "Left Hand")
        pose_landmarks = get_landmarks(resultant.pose_landmarks, "Pose")

        ##Almacenar los landmarks obtenidos en el archivo hd5f
        video_landmarks.create_dataset(f"Right_Hand_{count}", data=right_hand_landmarks)
        video_landmarks.create_dataset(f"Left_Hand_{count}", data=left_hand_landmarks)
        video_landmarks.create_dataset(f"Pose_{count}", data=pose_landmarks)

        count += 1


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

# Ruta al video
path = os.listdir('C:/Universidad/TFG/Desarrollo/ASL_videos/train/')
str_path = 'C:/Universidad/TFG/Desarrollo/ASL_videos/train/'

## Path para pequeña prueba
path_lil_test = os.listdir('C:/Universidad/TFG/Desarrollo/ASL_videos/lil_test/')
str_path_lil_test = 'C:/Universidad/TFG/Desarrollo/ASL_videos/lil_test/'


for selected_video in path_lil_test:
    ## Se crea un grupo por cada video
    
    video_landmarks = hd5f_file.create_group(selected_video) 
    
    ## Se llama a la función para procesar el video
    detectAll(mp.solutions.holistic.Holistic(static_image_mode=False, min_tracking_confidence=0.7, min_detection_confidence=0.7),
          str_path_lil_test + selected_video, draw=True, display=True)

## Se cierra el archivo al finalizar
hd5f_file.close()


# detectAll(mp.solutions.holistic.Holistic(static_image_mode=False, min_tracking_confidence=0.7, min_detection_confidence=0.7),
#           path_prueba, draw=True, display=True)

