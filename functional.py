from segmentation_models_pytorch.utils.meter                 import AverageValueMeter
from config                                                  import *
from tqdm                                                    import tqdm
import torch
import sys

def to_device_setting(device, loss, metrics):
    loss = loss.to(device)
    for metric in metrics:
        metric.to(device)
    return loss, metrics


def meter_setting(metrics):
    logs = {}
    loss_meter = AverageValueMeter()
    metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
    return logs, loss_meter, metrics_meters


def loss_update(logs, post_loss, loss, loss_meter):
    loss_value = post_loss.cpu().detach().numpy()
    loss_meter.add(loss_value)
    loss_logs = {'loss' : loss_meter.mean}
    logs.update(loss_logs)
    return logs, loss_meter


def metrics_update(logs, metrics, metrics_meter, y_pred, y):
    for metric in metrics:
        metric_value = metric(y_pred, y).cpu().detach().numpy()
        metrics_meter[metric.__name__].add(metric_value)
    metrics_logs = {k: v.mean for k, v in metrics_meter.items()}
    logs.update(metrics_logs)
    return logs, metrics_meter


def train_epoch(device, train_loader, model, loss, metrics, optimizer):
    model.train()
    loss, metrics = to_device_setting(device, loss, metrics)
    logs, loss_meter, metrics_meter = meter_setting(metrics)

    with tqdm(
        train_loader,
        desc='train',
        file=sys.stdout,
        disable=not (True),
    ) as iterator:
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred_t, y_pred_c = model.forward(x)[:,0,...], model.forward(x)[:,1,...]
            post_loss = 0.75*loss(y_pred_t, y[:,0,...]) + 0.25*loss(y_pred_c, y[:,1,...])
            post_loss.backward()
            optimizer.step()

            logs, loss_meter = loss_update(logs, post_loss, loss, loss_meter)

            logs, metrics_meter = metrics_update(logs, metrics, metrics_meter, y_pred_t, y[:,0,...])

            str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
            s = ', '.join(str_logs)
            iterator.set_postfix_str(s)
    return logs


def test_epoch(device, test_loader, model, loss, metrics, desc):
    model.eval()
    loss, metrics = to_device_setting(device, loss, metrics)
    logs, loss_meter, metrics_meter = meter_setting(metrics)

    with tqdm(
        test_loader,
        desc=desc,
        file=sys.stdout,
        disable=not (True),
    ) as iterator:
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_pred_t, y_pred_c = model.forward(x)[:,0,...], model.forward(x)[:,1,...]
                post_loss = 0.75*loss(y_pred_t, y[:,0,...]) + 0.25*loss(y_pred_c, y[:,1,...])

            logs, loss_meter = loss_update(logs, post_loss, loss, loss_meter)

            logs, metrics_meter = metrics_update(logs, metrics, metrics_meter, y_pred_t, y[:,0,...])

            str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
            s = ', '.join(str_logs)
            iterator.set_postfix_str(s)
    return logs

