# metrics.py
from sklearn import metrics as skmetrics
import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        # inicializar un diccionario de métricas
        self.metric_dict = {metric_name: [0] for metric_name in self.metric_names}

    def step(self, labels, preds):
        for metric in self.metric_names:
            # obtener la función de métrica
            do_metric = getattr(
                skmetrics, metric, "La métrica {} no está implementada".format(metric)
            )
            # verificar si la métrica requiere método de promedio, si es así establecer 'micro', 'macro' o 'None'
            try:
                self.metric_dict[metric].append(
                    do_metric(labels, preds, average="macro")
                )
            except:
                self.metric_dict[metric].append(do_metric(labels, preds))

    def epoch(self):
        # calcular métricas para una época completa
        avg = [sum(metric) / (len(metric) - 1) for metric in self.metric_dict.values()]
        metric_as_dict = dict(zip(self.metric_names, avg))
        return metric_as_dict

    def last_step_metrics(self):
        # retornar métricas de los últimos pasos
        values = [self.metric_dict[metric][-1] for metric in self.metric_names]
        metric_as_dict = dict(zip(self.metric_names, values))
        return metric_as_dict

def plot_learning_curves_advanced(train_losses, val_losses, train_accuracies, val_accuracies):
    """Graficar curvas de aprendizaje avanzadas con escalas uniformes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Gráfico de pérdidas con escala uniforme
    ax1.plot(epochs, train_losses, 'b-o', label='Entrenamiento', linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, 'r-s', label='Validación', linewidth=2, markersize=6)
    ax1.set_title('Evolución de la Pérdida Durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Pérdida', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Ajustar escala Y para pérdidas (usar rango más amplio para evitar variaciones exageradas)
    loss_min = min(min(train_losses), min(val_losses))
    loss_max = max(max(train_losses), max(val_losses))
    loss_range = loss_max - loss_min
    ax1.set_ylim(loss_min - loss_range * 0.1, loss_max + loss_range * 0.1)
    
    # Gráfico de precisión con escala uniforme
    ax2.plot(epochs, train_accuracies, 'g-o', label='Entrenamiento', linewidth=2, markersize=6)
    ax2.plot(epochs, val_accuracies, 'm-s', label='Validación', linewidth=2, markersize=6)
    ax2.set_title('Evolución de la Precisión Durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Precisión', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Ajustar escala Y para precisión (usar rango fijo más amplio)
    acc_min = min(min(train_accuracies), min(val_accuracies))
    acc_max = max(max(train_accuracies), max(val_accuracies))
    
    # Usar un rango mínimo del 10% para evitar variaciones exageradas
    if (acc_max - acc_min) < 0.1:
        center = (acc_max + acc_min) / 2
        ax2.set_ylim(center - 0.05, center + 0.05)
    else:
        acc_range = acc_max - acc_min
        ax2.set_ylim(acc_min - acc_range * 0.1, acc_max + acc_range * 0.1)
    
    plt.tight_layout()
    plt.show()


def plot_training_history_detailed(train_losses, val_losses, train_accuracies, val_accuracies):
    """Graficar historial detallado de entrenamiento con escalas mejoradas"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Pérdida de entrenamiento
    axes[0, 0].plot(epochs, train_losses, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Pérdida de Entrenamiento', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Pérdida')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pérdida de validación
    axes[0, 1].plot(epochs, val_losses, 'r-s', linewidth=2, markersize=4)
    axes[0, 1].set_title('Pérdida de Validación', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Pérdida')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precisión de entrenamiento
    axes[1, 0].plot(epochs, train_accuracies, 'g-o', linewidth=2, markersize=4)
    axes[1, 0].set_title('Precisión de Entrenamiento', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Precisión')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precisión de validación
    axes[1, 1].plot(epochs, val_accuracies, 'm-s', linewidth=2, markersize=4)
    axes[1, 1].set_title('Precisión de Validación', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Precisión')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Ajustar escalas para cada subplot individualmente
    for i in range(2):
        for j in range(2):
            if i == 0:  # Gráficos de pérdida
                data = train_losses if j == 0 else val_losses
                data_min, data_max = min(data), max(data)
                data_range = data_max - data_min
                axes[i, j].set_ylim(data_min - data_range * 0.1, data_max + data_range * 0.1)
            else:  # Gráficos de precisión
                data = train_accuracies if j == 0 else val_accuracies
                data_min, data_max = min(data), max(data)
                
                # Usar rango mínimo del 5% para precisión individual
                if (data_max - data_min) < 0.05:
                    center = (data_max + data_min) / 2
                    axes[i, j].set_ylim(center - 0.025, center + 0.025)
                else:
                    data_range = data_max - data_min
                    axes[i, j].set_ylim(data_min - data_range * 0.1, data_max + data_range * 0.1)
    
    plt.tight_layout()
    plt.show()