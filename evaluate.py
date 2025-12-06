import torch
import numpy as np
import logging

#CI Helper Function
def get_cindex(y_true, y_pred):
    g = y_true.flatten()
    p = y_pred.flatten()
    summ = 0
    pair = 0
    for i in range(1, len(g)):
        for j in range(0, i):
            if i is not j:
                if(g[i] > g[j]):
                    pair += 1
                    summ +=  1* (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    if pair != 0:
        return summ/pair
    else:
        return 0

#CI, MSE, RMSE
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    ci = get_cindex(y_true, y_pred)
    
    return {'mse': mse, 'rmse': rmse, 'ci': ci}

#Computes val loss, returns metrics, true labels, and predictions
def evaluate(model, loss_fn, dataloader, params, model_type='baseline'):
    model.eval()
    losses = []

    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if model_type == 'baseline':
                pr, lig, y = batch
                if params.cuda:
                    pr, lig, y = pr.cuda(non_blocking=True), lig.cuda(non_blocking=True), y.cuda(non_blocking=True)
                pred = model(pr, lig)
            else:
                pr, lig, stereo, desc, y = batch
                if params.cuda:
                    pr, lig, stereo, desc, y = pr.cuda(non_blocking=True), lig.cuda(non_blocking=True), \
                                               stereo.cuda(non_blocking=True), desc.cuda(non_blocking=True), \
                                               y.cuda(non_blocking=True)
                pred = model(pr, lig, stereo, desc)

            loss = loss_fn(pred, y)
            losses.append(loss.item())

            total_preds.append(pred.cpu().numpy())
            total_labels.append(y.cpu().numpy())

    avg_loss = np.mean(losses)

    final_preds = np.concatenate(total_preds).flatten()
    final_labels = np.concatenate(total_labels).flatten()

    metrics = calculate_metrics(final_labels, final_preds)
    metrics['loss'] = avg_loss # 
    
    logging.info(f"- Validation loss: {avg_loss:.4f}")
    
    return metrics, final_labels, final_preds