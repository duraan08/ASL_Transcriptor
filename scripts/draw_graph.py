import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import datetime
from matplotlib.lines import Line2D

count = 1
def drawLossGraphic(loss_values, accuracy_values_test, num_epochs, dateTime, hyperparameters):
    plt.figure()
    plt.plot(loss_values, linestyle='-', label='Función de Pérdida', color='blue')

    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')

    plt.xscale('log')

    custom_lines = [Line2D([0], [0], color='blue', lw=2)]
    plt.legend(custom_lines, ['Función de Pérdida'], loc='lower left')

    # Agrega los hiperparámetros como texto en la parte superior del gráfico
    plt.text(0.01, 0.95, hyperparameters, transform=plt.gcf().transFigure, fontsize=10, verticalalignment='top')

    path_fichero = f"/scratch/uduran005/tfg-workspace/graphics/loss/grafica_perdida_{dateTime}_1.pdf"
    path_general = f"/scratch/uduran005/tfg-workspace/graphics/loss"
    
    if (not os.path.exists(path_fichero)):
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/loss/grafica_perdida_{dateTime}_1.pdf")     
    else:
        file_list = os.listdir(path_general)
        for file in file_list:
            file_index = max(file.split("_")[3].split(".")[0])
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/loss/grafica_perdida_{dateTime}_{int(file_index) + 1}.pdf")     

def drawEvalGraphic(accuracy_values_train, accuracy_values_test, num_epochs, dateTime, hyperparameters):
    plt.figure()
    plt.plot(accuracy_values_train, linestyle='-', label='Accuracy train', color='red')
    plt.plot(accuracy_values_test, linestyle='-', label='Accuracy evaluación', color='blue')

    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')

    plt.xscale('log')

    custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='blue', lw=2)]
    plt.legend(custom_lines, ['Accuracy train', 'Accuracy evalucación'], loc='upper left')

    # Agrega los hiperparámetros como texto en la parte superior del gráfico
    plt.text(0.01, 0.95, hyperparameters, transform=plt.gcf().transFigure, fontsize=10, verticalalignment='top')


    path_fichero = f"/scratch/uduran005/tfg-workspace/graphics/accuracy/grafica_accuracy_{dateTime}_1.pdf"
    path_general = f"/scratch/uduran005/tfg-workspace/graphics/accuracy"
    
    if (not os.path.exists(path_fichero)):
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/accuracy/grafica_accuracy_{dateTime}_1.pdf")     
    else:
        file_list = os.listdir(path_general)
        for file in file_list:
            file_index = max(file.split("_")[3].split(".")[0])
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/accuracy/grafica_accuracy_{dateTime}_{int(file_index) + 1}.pdf")     

def drawGraph(values, values2, epochs, datetime, hidden_dim, num_layers, num_heads, learning_rate, batch_size, weight_decay, ty):
    
    hyperparameters = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }
    hyperparameters_string = '  '.join([f'{key}: {value}' for key, value in hyperparameters.items()])

    if (ty == "acc"):
        drawEvalGraphic(values, values2, epochs, datetime, hyperparameters_string)
    elif (ty == 'loss'):
        drawLossGraphic(values, values2, epochs, datetime, hyperparameters_string)
