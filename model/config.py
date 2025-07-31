################################################################################
# filename: config.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 31/07,2025
################################################################################

import os
import json
from argparse import Namespace

import torch

from model.model import ClassificatorCVTransformer
from model.tokenizer import TokenizerCVTransformer

################################################################################

class ConfigTraining:
    def __init__(self, args : Namespace):
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        self.epoch = 0
        self.old_accuracy = 0

    def save_file(self, epoch : int, model : torch.nn.Module, optimizer : torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau):
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        directory_checkpoint = os.path.join(self.models_path, f"checkpoint_{epoch}")
        self.checkpoint_directory = self.models_path + f"/checkpoint_{epoch}"
        if not os.path.exists(directory_checkpoint):
            os.makedirs(directory_checkpoint)
        path_save = os.path.join(self.models_path, f"checkpoint_{epoch}")
        torch.save(model.state_dict(), os.path.join(path_save, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(path_save, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(path_save, "scheduler.pt"))
        self.epoch = epoch
        with open(os.path.join(path_save, "config.json"), "w") as f:
            json.dump(self.__dict__, f)
    
    def load_state_training(self, args : Namespace):
        self.epoch = args.epoch
        self.checkpoint_directory = args.models_path + f"/checkpoint_{self.epoch}"
        if not os.path.exists(self.checkpoint_directory):
            raise Exception(f"Checkpoint directory {self.checkpoint_directory} does not exist")
        
        with open(self.checkpoint_directory + "/config.json", "r") as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(self, k, v)
        device = torch.device(self.device)
        
        model = ClassificatorCVTransformer(self.model).to(device)
        model.load_state_dict(torch.load(self.checkpoint_directory + "/model.pt"))
        tokenizer = TokenizerCVTransformer(self.model)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.lr))
        optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint_directory, "optimizer.pt")))
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
        scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_directory, "scheduler.pt")))

        return model, tokenizer, optimizer, scheduler

################################################################################
# End of File
################################################################################