# metrics.py
from sklearn import metrics as skmetrics
import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        # initialize a metric dictionary
        self.metric_dict = {metric_name: [0] for metric_name in self.metric_names}

    def step(self, labels, preds):
        for metric in self.metric_names:
            # get the metric function
            do_metric = getattr(
                skmetrics, metric, "The metric {} is not implemented".format(metric)
            )
            # check if metric require average method, if yes set to 'micro' or 'macro' or 'None'
            try:
                self.metric_dict[metric].append(
                    do_metric(labels, preds, average="macro")
                )
            except:
                self.metric_dict[metric].append(do_metric(labels, preds))

    def epoch(self):
        # calculate metrics for an entire epoch
        avg = [sum(metric) / (len(metric) - 1) for metric in self.metric_dict.values()]
        metric_as_dict = dict(zip(self.metric_names, avg))
        return metric_as_dict

    def last_step_metrics(self):
        # return metrics of last steps
        values = [self.metric_dict[metric][-1] for metric in self.metric_names]
        metric_as_dict = dict(zip(self.metric_names, values))
        return metric_as_dict

def plot_learning_curves_advanced(train_losses, val_losses, train_accuracies, val_accuracies):
    """Crear curvas de aprendizaje más profesionales con matplotlib"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    
    # 1. Curva de pérdida con área sombreada
    ax1.plot(epochs, train_losses, 'b-', label='Entrenamiento', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validación', linewidth=2)
    ax1.fill_between(epochs, train_losses, alpha=0.3, color='blue')
    ax1.fill_between(epochs, val_losses, alpha=0.3, color='red')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Curva de Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Curva de accuracy con bandas de confianza
    ax2.plot(epochs, train_accuracies, 'g-', label='Entrenamiento', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'm-', label='Validación', linewidth=2)
    ax2.fill_between(epochs, train_accuracies, alpha=0.3, color='green')
    ax2.fill_between(epochs, val_accuracies, alpha=0.3, color='magenta')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Curva de Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # Estadísticas adicionales
    print("\n" + "="*50)
    print("ESTADÍSTICAS DE ENTRENAMIENTO")
    print("="*50)
    print(f"Mejor accuracy de entrenamiento: {max(train_accuracies):.4f}")
    print(f"Mejor accuracy de validación: {max(val_accuracies):.4f}")
    print(f"Pérdida final de entrenamiento: {train_losses[-1]:.4f}")
    print(f"Pérdida final de validación: {val_losses[-1]:.4f}")
    print("="*50)


def plot_training_history_detailed(train_losses, val_losses, train_accuracies, val_accuracies):
    """Visualización detallada del historial de entrenamiento"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Gráfico de pérdida
    ax1.plot(epochs, train_losses, 'b-o', label='Entrenamiento', markersize=4, linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Validación', markersize=4, linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Evolución de la Pérdida Durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de accuracy
    ax2.plot(epochs, train_accuracies, 'g-o', label='Entrenamiento', markersize=4, linewidth=2)
    ax2.plot(epochs, val_accuracies, 'm-s', label='Validación', markersize=4, linewidth=2)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Evolución de la Precisión Durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()