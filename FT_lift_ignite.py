from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from dataset.sun_data_loader import GetLoader
from loss.LiftedStructure import LiftedStructureLoss
from models.lift_model import CNNModel
try:
    import visdom
except ImportError:
    raise RuntimeError("No visdom package is found. Please install it with command: \n pip install visdom")
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision
from ignite.handlers import ModelCheckpoint



cuda = True
dset_classes_number = 19
model_ft = CNNModel()
if cuda:
    model_ft = model_ft.cuda()
criterion = {'label':nn.NLLLoss(),
            'structured':LiftedStructureLoss()
}

for idx, m in enumerate(model_ft.named_modules()):
    print(idx, '-->', m)
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
ignored_params = list(map(id, model_ft.feature.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model_ft.parameters())
# optimizer_ft = optim.SGD([
#             {'params': base_params},
#             {'params': model_ft.classifier.parameters(), 'lr': 1e-3}
#             ],lr=0.0001, momentum=0.9)

optimizer_ft = optim.Adam([
            {'params': base_params},
            {'params': model_ft.feature.classifier.parameters(), 'lr': 1e-3}
            ],lr=1e-4, betas=(0.9, 0.99), eps=1e-08, weight_decay=0)



exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load data
batch_size = 256
image_size = 256
#load image 
source_image_root = os.path.join('/1116', 'SUN', 'rgb')
target_image_root = os.path.join('/1116', 'SUN', 'hha')
train_list = os.path.join('/1116', 'SUN', 'train_label.txt')
test_list = os.path.join('/1116', 'SUN', 'test_label.txt')
#phase = 'train' # train or test
data_list = {
    'train':train_list,
    'test':test_list
}

domain = 'source' #source or target
data_image_root = {'source':source_image_root, 
                    'target':target_image_root
}

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.70183206, 0.33803278, 0.4370506 ], [0.23395756, 0.2259527, 0.18689048])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.70183206, 0.33803278, 0.4370506 ], [0.23395756, 0.2259527, 0.18689048])
    ]),
}


dataset ={phase: GetLoader(
    data_root=os.path.join(data_image_root[domain], phase),
    data_list=data_list[phase],
    transform=data_transforms[phase])
    for phase in ['train', 'test']
}
dataset_sizes = {phase: len(dataset[phase]) for phase in ['train', 'test']}
dataloaders ={phase: torch.utils.data.DataLoader(
    dataset=dataset[phase],
    batch_size=batch_size,
    shuffle=False,
    num_workers=8)
    for phase in ['train', 'test']}


def create_plot_window(vis, xlabel, ylabel, title):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))


def run(model, criterion, optimizer, epochs=100, log_interval=10):
    vis = visdom.Visdom(env='ft_lift_ignite')

    train_loader = dataloaders['train']
    val_loader = dataloaders['test']

    # if not vis.check_connection():
    #     raise RuntimeError("Visdom server not running. Please run python -m visdom.server")

    # trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    # evaluator = create_supervised_evaluator(model,
    #                                         metrics={'accuracy': Accuracy(criterion['label']),
    #                                                  'nll': Loss(criterion['label']),
    #                                                  'precision': Precision(average=True )},
    #                                         device=device)

    def update_model(trainer, batch):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        #inputs, labels = _prepare_batch(batch, device=device)
        optimizer.zero_grad()
        class_output, structured_output = model(inputs)
        loss = criterion['label'](class_output, labels)+criterion['structured'](structured_output, labels)
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 
                 'class_output': class_output, 
                 'structured_output': structured_output, 
                 #'inputs': inputs, 
                 'labels': labels}
    trainer = Engine(update_model)

    # def _prepare_batch(batch, device=None, non_blocking=False):
    #     """Prepare batch for training: pass to a device with options

    #     """
    #     x, y = batch
    #     return (convert_tensor(x, device=device, non_blocking=non_blocking),
    #             convert_tensor(y, device=device, non_blocking=non_blocking))

    def _inference(evaluator, batch):
        model.eval()
        with torch.no_grad():
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            class_output, structured_output = model(inputs)
            loss = criterion['label'](class_output, labels)+criterion['structured'](structured_output, labels)
            return {'loss': loss.item(), 
                 'class_output': class_output, 
                 'structured_output': structured_output, 
                 #'inputs': inputs, 
                 'labels': labels}
    evaluator = Engine(_inference)

    output_transform1 = lambda data: (data['class_output'], data['labels'])
    output_transform2 = lambda data: (data['structured_output'], data['labels'])

    metric_accuracy = Accuracy(output_transform=output_transform1)
    metric_accuracy.attach(evaluator, 'accuracy')

    metric_nll = Loss(criterion['label'], output_transform=output_transform1)
    metric_nll.attach(evaluator, 'nll')

    metric_precision = Precision(average=True, output_transform=output_transform1)
    metric_precision.attach(evaluator, 'precision')

    # evaluator = create_supervised_evaluator(model,
    #                                     metrics={'accuracy': Accuracy(output_transform=output_transform1),
    #                                                 'nll': Loss(criterion['label'], output_transform=output_transform1),
    #                                                 'precision': Precision(average=True, output_transform=output_transform1)},
    #                                     device=device)

    handler = ModelCheckpoint('/1116/tmp/lift_models', 'myprefix', save_interval=1, n_saved=150, require_empty=False, create_dir=True)

    train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss')
    train_avg_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Average Loss')
    train_avg_accuracy_window = create_plot_window(vis, '#Iterations', 'Accuracy', 'Training Average Accuracy')
    train_avg_precision_window = create_plot_window(vis, '#Iterations', 'Precision', 'Training Average Precision')
    val_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Validation Average Loss')
    val_avg_accuracy_window = create_plot_window(vis, '#Epochs', 'Accuracy', 'Validation Average Accuracy')
    val_avg_precision_window = create_plot_window(vis, '#Epochs', 'Precision', 'Validation Average Precison')
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output['loss']))
            vis.line(X=np.array([engine.state.iteration]),
                     Y=np.array([engine.state.output['loss']]),
                     update='append', win=train_loss_window)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        avg_precision = metrics['precision']
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Avg Precision: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll, avg_precision))
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]),
                 win=train_avg_accuracy_window, update='append')
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_nll]),
                 win=train_avg_loss_window, update='append')
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_precision]),
                 win=train_avg_precision_window, update='append')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        avg_precision = metrics['precision']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Avg Precision: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll, avg_precision))
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]),
                 win=val_avg_accuracy_window, update='append')
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_nll]),
                 win=val_avg_loss_window, update='append')
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_precision]),
                 win=val_avg_precision_window, update='append')        
                 
    
    # kick everything off
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'mymodel': model})
    trainer.run(train_loader, max_epochs=epochs)


run(model_ft, criterion, optimizer_ft, epochs=500, log_interval=10)