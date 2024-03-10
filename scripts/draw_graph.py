import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import datetime
from matplotlib.lines import Line2D

count = 1
def drawLossGraphic(loss_values, loss_values_test, num_epochs, dateTime, hyperparameters):
    plt.figure()
    plt.plot(loss_values, linestyle='-', label='Loss train', color='red')
    plt.plot(loss_values_test, linestyle='-', label='Loss evaluación', color='green')


    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')

    plt.xscale('log')

    custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='green', lw=2)]
    plt.legend(custom_lines, ['Loss train', 'Loss evaluación'], loc='lower left')

    # Agrega los hiperparámetros como texto en la parte superior del gráfico
    plt.text(0.01, 0.95, hyperparameters, transform=plt.gcf().transFigure, fontsize=9, verticalalignment='top')

    path_fichero_loss = f"/scratch/uduran005/tfg-workspace/graphics/loss/grafica_perdida_{dateTime}_1.pdf"
    path_general_loss = f"/scratch/uduran005/tfg-workspace/graphics/loss"
    indices_loss = [] 
    
    if (not os.path.exists(path_fichero_loss)):
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/loss/grafica_perdida_{dateTime}_1.pdf")     
    else:
        file_list = os.listdir(path_general_loss)
        for file in file_list:
            if (file.split('_')[2] == dateTime):
                file_index_loss = file.split('_')[3].split('.')[0]
                indices_loss.append(int(file_index_loss))
        print(f"Maximo indice loss --> {max(indices_loss)}")
        max_index_loss = max(indices_loss)
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/loss/grafica_perdida_{dateTime}_{max_index_loss + 1}.pdf")     

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
    plt.text(0.01, 0.95, hyperparameters, transform=plt.gcf().transFigure, fontsize=9, verticalalignment='top')


    path_fichero_acc = f"/scratch/uduran005/tfg-workspace/graphics/accuracy/grafica_accuracy_{dateTime}_1.pdf"
    path_general_acc = f"/scratch/uduran005/tfg-workspace/graphics/accuracy"
    indices_acc = []
    
    if (not os.path.exists(path_fichero_acc)):
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/accuracy/grafica_accuracy_{dateTime}_1.pdf")     
    else:
        file_list = os.listdir(path_general_acc)
        for file in file_list:
            if (file.split('_')[2] == dateTime):
                file_index_acc = file.split('_')[3].split('.')[0]
                indices_acc.append(int(file_index_acc))
        print(f"Max index acc --> {max(indices_acc)}")
        max_index_acc = max(indices_acc)
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/accuracy/grafica_accuracy_{dateTime}_{max_index_acc + 1}.pdf")    

def drawTop5Graphic(accuracy_values_top5, accuracy_values_top1, num_epochs, dateTime, hyperparameters):
    plt.figure()
    plt.plot(accuracy_values_top5, linestyle='-', label='Accuracy Top-5', color='red')
    plt.plot(accuracy_values_top1, linestyle='-', label='Accuracy Top-1', color='blue')

    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')

    plt.xscale('log')

    custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='blue', lw=2)]
    plt.legend(custom_lines, ['Accuracy Top-5', 'Accuracy Top-1'], loc='upper left')

    # Agrega los hiperparámetros como texto en la parte superior del gráfico
    plt.text(0.01, 0.95, hyperparameters, transform=plt.gcf().transFigure, fontsize=9, verticalalignment='top')


    path_fichero = f"/scratch/uduran005/tfg-workspace/graphics/accuracy-top/grafica_accuracyTop_{dateTime}_1.pdf"
    path_general = f"/scratch/uduran005/tfg-workspace/graphics/accuracy-top"
    indices_top5 = []
    
    if (not os.path.exists(path_fichero)):
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/accuracy-top/grafica_accuracyTop_{dateTime}_1.pdf")     
    else:
        file_list = os.listdir(path_general)
        for file in file_list:
            if (file.split('_')[2] == dateTime):
                file_index = file.split('_')[3].split('.')[0]
                indices_top5.append(int(file_index))
        print(f"Max index top5 --> {max(indices_top5)}")
        max_index_top5 = max(indices_top5)
        plt.savefig(f"/scratch/uduran005/tfg-workspace/graphics/accuracy-top/grafica_accuracyTop_{dateTime}_{max_index_top5 + 1}.pdf")  


def drawGraph(values, values2, epochs, datetime, hidden_dim, num_layers, num_heads, learning_rate, batch_size, weight_decay, dropout, ty):
    
    hyperparameters = {
        'Hidden dim': hidden_dim,
        'Layers': num_layers,
        'Heads': num_heads,
        'LR': learning_rate,
        'Batch': batch_size,
        'WD': weight_decay,
        'Dropout': dropout
    }
    hyperparameters_string = '  '.join([f'{key}: {value}' for key, value in hyperparameters.items()])

    if (ty == 'acc'):
        drawEvalGraphic(values, values2, epochs, datetime, hyperparameters_string)
    elif (ty == 'loss'):
        drawLossGraphic(values, values2, epochs, datetime, hyperparameters_string)
    elif (ty == 'top5'):
        drawTop5Graphic(values, values2, epochs, datetime, hyperparameters_string)
