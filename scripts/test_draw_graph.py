import matplotlib.pyplot as plt
import os
import datetime

count = 1
def drawLossGraphic(loss_values, num_epochs, dateTime):
    # Crear un gráfico de la función de pérdida con puntos resaltados
    plt.figure(figsize=(len(loss_values), num_epochs))
    plt.plot(loss_values, marker='o', linestyle='-', label='Función de Pérdida', color='blue')

    # Añadir puntos de la función de pérdida
    plt.scatter(range(len(loss_values)), loss_values, color='red', s=50, label='Puntos')

    # Títulos y etiquetas
    plt.title('Función de Pérdida del Transformer Encoder')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Mostrar el gráfico
    # plt.grid(True)
    # plt.show()

    path_fichero = f"C:/Universidad/TFG/Desarrollo/Graphics/Loss/grafica_perdida_{dateTime}_1.pdf"
    path_general = f"C:/Universidad/TFG/Desarrollo/Graphics/Loss"
    
    if (not os.path.exists(path_fichero)):
        plt.savefig(f"C:/Universidad/TFG/Desarrollo/Graphics/Loss/grafica_perdida_{dateTime}_1.pdf")     ##Guardar la grafica en .pdf
    else:
        file_list = os.listdir(path_general)
        for file in file_list:
            file_index = max(file.split("_")[3].split(".")[0])
        plt.savefig(f"C:/Universidad/TFG/Desarrollo/Graphics/Loss/grafica_perdida_{dateTime}_{int(file_index) + 1}.pdf")     ##Guardar la grafica en .pdf


def drawEvalGraphic(accuracy_values, num_epochs, dateTime):
    # Crear un gráfico de la función de pérdida con puntos resaltados
    plt.figure(figsize=(len(accuracy_values), num_epochs))
    plt.plot(accuracy_values, marker='o', linestyle='-', label='Grafica de precisión', color='red')

    # Añadir puntos de la función de pérdida
    plt.scatter(range(len(accuracy_values)), accuracy_values, color='blue', s=50, label='Puntos')

    # Títulos y etiquetas
    plt.title('Grafica de precisión del Transformer Encoder')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    # Mostrar el gráfico
    # plt.grid(True)
    # plt.show()

    path_fichero = f"C:/Universidad/TFG/Desarrollo/Graphics/Accuracy/grafica_accuracy_{dateTime}_1.pdf"
    path_general = f"C:/Universidad/TFG/Desarrollo/Graphics/Accuracy"
    
    if (not os.path.exists(path_fichero)):
        plt.savefig(f"C:/Universidad/TFG/Desarrollo/Graphics/Accuracy/grafica_accuracy_{dateTime}_1.pdf")     ##Guardar la grafica en .pdf
    else:
        file_list = os.listdir(path_general)
        for file in file_list:
            file_index = max(file.split("_")[3].split(".")[0])
        
        plt.savefig(f"C:/Universidad/TFG/Desarrollo/Graphics/Accuracy/grafica_accuracy_{dateTime}_{int(file_index) + 1}.pdf")     ##Guardar la grafica en .pdf

