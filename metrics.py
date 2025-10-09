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

def plot_training_history_detailed(train_losses, val_losses, train_accuracies, val_accuracies):
    """Visualización detallada del historial de entrenamiento"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Gráfico de pérdida
    ax1.plot(epochs, train_losses, 'b-o', label='Entrenamiento', markersize=4, linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Validación', markersize=4, linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    
    # Ajustar escala Y para pérdidas - rango más amplio
    loss_min = min(min(train_losses), min(val_losses))
    loss_max = max(max(train_losses), max(val_losses))
    loss_range = loss_max - loss_min
    ax1.set_ylim(loss_min - loss_range * 0.4, loss_max + loss_range * 0.4)
    
    ax1.set_title('Evolución de la Pérdida Durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de accuracy
    ax2.plot(epochs, train_accuracies, 'g-o', label='Entrenamiento', markersize=4, linewidth=2)
    ax2.plot(epochs, val_accuracies, 'm-s', label='Validación', markersize=4, linewidth=2)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    
    # Ajustar escala Y para accuracy - rango más amplio
    acc_min = min(min(train_accuracies), min(val_accuracies))
    acc_max = max(max(train_accuracies), max(val_accuracies))
    acc_range = acc_max - acc_min
    # Ampliar el rango para accuracy, manteniendo límites lógicos
    y_min = max(0, acc_min - acc_range * 0.25)
    y_max = min(1, acc_max + acc_range * 0.25)
    ax2.set_ylim(y_min, y_max)
    
    ax2.set_title('Evolución de la Precisión Durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Estadísticas adicionales
    print("\n" + "="*50)
    print("ESTADÍSTICAS DE ENTRENAMIENTO")
    print("="*50)
    print(f"Accuracy de entrenamiento: {max(train_accuracies):.4f}")
    print(f"Accuracy de validación: {max(val_accuracies):.4f}")
    print(f"Pérdida de entrenamiento: {train_losses[-1]:.4f}")
    print(f"Pérdida de validación: {val_losses[-1]:.4f}")
    print("="*50)