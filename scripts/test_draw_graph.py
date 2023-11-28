import matplotlib.pyplot as plt

# Suponiendo que tienes una lista o array de valores de pérdida
loss_values = [0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01]

# Crear un gráfico de la función de pérdida con puntos resaltados
plt.figure(figsize=(10, 6))
plt.plot(loss_values, marker='o', linestyle='-', label='Función de Pérdida', color='blue')

# Añadir puntos de la función de pérdida
plt.scatter(range(len(loss_values)), loss_values, color='red', s=50, label='Puntos')

# Títulos y etiquetas
plt.title('Función de Pérdida del Transformer Encoder')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()
