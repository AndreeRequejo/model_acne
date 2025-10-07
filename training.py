# training.py
import torch
import torch.nn as nn
import numpy as np

def train_one_epoch(
    model,
    train_loader,
    test_loader,
    device,
    optimizer,
    criterion,
    train_metrics,
    val_metrics,
):
    # entrenar-el-modelo
    train_loss = 0
    valid_loss = 0
    all_labels = []
    all_preds = []
    model.train()
    for data, target in train_loader:
        # mover-tensores-a-GPU
        data = data.type(torch.FloatTensor).to(device)
        target = target.float().to(device)
        # limpiar-gradientes-de-todas-las-variables-optimizadas
        optimizer.zero_grad()
        # paso-hacia-adelante: calcular-salidas-predichas-pasando-entradas-al-modelo
        output = model(data)
        preds = torch.argmax(output, axis=1).cpu().detach().numpy()
        labels = target.cpu().numpy()
        # calcular-la-pérdida-del-lote
        loss = criterion(output.type(torch.FloatTensor), target.type(torch.LongTensor))
        # paso-hacia-atrás: calcular-gradiente-de-la-pérdida-respecto-a-parámetros-del-modelo
        loss.backward()
        # realizar-un-paso-de-optimización (actualización-de-parámetros)
        optimizer.step()
        # actualizar-pérdida-de-entrenamiento
        train_loss += loss.item() * data.size(0)
        # calcular métricas de entrenamiento
        all_labels.extend(labels)
        all_preds.extend(preds)
    
    train_metrics.step(all_labels, all_preds)

    # validar-el-modelo
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.type(torch.FloatTensor).to(device)
            target = target.to(device)
            output = model(data)
            preds = torch.argmax(output, axis=1).tolist()
            labels = target.tolist()
            all_labels.extend(labels)
            all_preds.extend(preds)
            loss = criterion(output, target)

            # actualizar-pérdida-promedio-de-validación
            valid_loss += loss.item() * data.size(0)

    val_metrics.step(all_labels, all_preds)
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(test_loader.sampler)

    return (
        train_loss,
        valid_loss,
        train_metrics.last_step_metrics(),
        val_metrics.last_step_metrics(),
    )


def test_result(model, test_loader, device, name='no_tta_prob.npy'):
    """Probar el modelo cambiando a modo de evaluación"""
    model.eval()
    preds = []
    aprobs = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            # mover-tensores-a-GPU
            data = data.to(device)
            # paso-hacia-adelante: calcular-salidas-predichas-pasando-entradas-al-modelo
            output = model(data)
            prob = nn.Softmax(dim=1)
            # aplicar Softmax a los resultados
            probs = prob(output)
            aprobs.append(probs.cpu())
            labels.append(target.cpu().numpy())
            preds.extend(torch.argmax(probs, axis=1).tolist())
    aprobs = np.array(aprobs)
    np.save(name, aprobs)
    return preds, np.array(labels)