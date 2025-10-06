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
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
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


def plot_training_history_detailed(train_losses, val_losses, train_accuracies, val_accuracies):
    """Visualización detallada del historial de entrenamiento"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Layout más complejo
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1])
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Gráfico principal - pérdida
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, train_losses, 'b-o', label='Entrenamiento', markersize=4, linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Validación', markersize=4, linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Evolución de la Pérdida Durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico principal - accuracy
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(epochs, train_accuracies, 'g-o', label='Entrenamiento', markersize=4, linewidth=2)
    ax3.plot(epochs, val_accuracies, 'm-s', label='Validación', markersize=4, linewidth=2)
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Evolución de la Precisión Durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()