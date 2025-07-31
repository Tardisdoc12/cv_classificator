################################################################################
# filename: train.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 31/07,2025
################################################################################

import copy

import torch
from tqdm import tqdm

from model.config import ConfigTraining

################################################################################

threshold_allowed = 0.9

################################################################################

def update_model_ema(
    model : torch.nn.Module,
    ema_model : torch.nn.Module,
    alpha : int = 0.9999
):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

################################################################################

@torch.no_grad()
def evaluate_model(
    model : torch.nn.Module,
    dataloader_validation : torch.utils.data.DataLoader,
    device : torch.device = "gpu"
):
    model.eval()
    total = 0
    correct = 0
    for batch in dataloader_validation:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        scores_predicted = model(input_ids, attention_mask)
        rounded_preds = torch.round(scores_predicted).clamp(0, 5)
        correct += (rounded_preds == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    return accuracy

################################################################################

def validation_epoch(
    epoch : int, 
    model : torch.nn.Module,
    config: ConfigTraining,
    dataloader_val : torch.utils.data.DataLoader,
    device : torch.device
):
    stop = False
    if epoch % config.interval_valid == 0:
        accuracy = evaluate_model(model, dataloader_val, device)
        with open("accuracy.txt", "a") as f:
            f.write(f"Epoch {epoch}: {accuracy}\n")
        if accuracy > threshold_allowed:
            torch.save(model.state_dict(), "models/best_model_over_threshold.pt")
            stop = True
        model.train()
    return stop

################################################################################

def training_epoch(
    model : torch.nn.Module,
    ema_model: torch.nn.Module,
    config: ConfigTraining,
    dataloader_train : torch.utils.data.DataLoader,
    optimizer : torch.optim.Optimizer,
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau,
    device : torch.device
):
    total = 0
    correct = 0
    for batch in dataloader_train:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        scores_predicted = model(input_ids, attention_mask)
        
        loss = torch.nn.functional.mse_loss(scores_predicted, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_model_ema(model, ema_model)

        rounded_preds = torch.round(scores_predicted).clamp(0, 5)
        correct += (rounded_preds == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    scheduler.step(accuracy)
    config.save_file(config.epoch, model, optimizer, config.old_accuracy, scheduler)

    if accuracy > config.old_accuracy:
        config.old_accuracy = accuracy
        torch.save(model.state_dict(), config.checkpoint_directory + "/best_model.pt")
    return model

################################################################################

def training_loop(
    model,
    config,
    dataloader_train_val,
    optimizer,
    scheduler
):
    device = torch.device(config.device)
    epoches = config.epoches
    evaluate = config.evaluate
    model.to(device)
    model.train()

    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False
    
    for epoch in tqdm(range(config.epoch, epoches), unit="epoch"):
        if validation_epoch(epoch, ema_model, config, dataloader_train_val["validation"], device):
            break
        model = training_epoch(model, dataloader_train_val["train"], optimizer, scheduler, device)
    if evaluate:
        evaluate_model = evaluate_model(model, dataloader_train_val["test"], device)
        print(f"Accuracy on test set: {evaluate_model}")

################################################################################

def evaluate_model(
    model : torch.nn.Module,
    dataloader_test : torch.utils.data.DataLoader,
    device : torch.device = ("cuda" if torch.cuda.is_available() else "cpu")
):
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    for batch in dataloader_test:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        scores_predicted = model(input_ids, attention_mask)
        rounded_preds = torch.round(scores_predicted).clamp(0, 5)
        correct += (rounded_preds == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    return accuracy

################################################################################
# End of File
################################################################################