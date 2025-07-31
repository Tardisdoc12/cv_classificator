################################################################################
# filename: main_train.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 31/07,2025
################################################################################

import os

import yaml
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from training.train import training_loop
from model.model import ClassificatorCVTransformer
from model.tokenizer import TokenizerCVTransformer
from model.dataset_creater import create_dataloader
from model.config import ConfigTraining

################################################################################

def get_arg_parser() -> Namespace:
    argparser = ArgumentParser(
        prog="main_train.py",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument("--epoches", type=int, default=10_000, help="Number of epoches")
    argparser.add_argument("--device", choices=["cpu", "cuda"],type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    argparser.add_argument("--model", type=str, default="prajjwal1/bert-tiny", help="Model to use")
    argparser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    argparser.add_argument("--interval_valid", type=int, default=100, help="Interval between validation")
    argparser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    argparser.add_argument("--max_length", type=int, default=512, help="Max length of the input")
    argparser.add_argument("--datas_path", type=str, default="datas_for_training/datas_dict.py", help="Path to the datas dict")
    argparser.add_argument("--models_path", type=str, default="models/", help="Path to save the models")
    argparser.add_argument("--evaluate", type=bool, default=True, help="If True, evaluate the model after training")
    argparser.add_argument("--restart", type=bool, default=True, help="If True, restart from the epoch specified in epoch")
    argparser.add_argument("epoch", type=int, default=0, help="Epoch to restart from")
    return argparser.parse_args()

################################################################################

def main():
    args : Namespace = get_arg_parser()
    config = ConfigTraining(args)
    device = torch.device(args.device)
    
    if not os.path.exists(args.models_path):
        os.makedirs(args.models_path)
    
    if args.restart:
        model, tokenizer, optimizer, scheduler = config.load_state_training(args)
        datas_dict = {}
        with open(config.datas_path, "r") as f:
            datas_dict = yaml.safe_load(f)
        datas_dict = datas_dict["datas_dict"]
    else:
        datas_dict = {}
        with open(args.datas_path, "r") as f:
            datas_dict = yaml.safe_load(f)
        datas_dict = datas_dict["datas_dict"]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
        model = ClassificatorCVTransformer(args.model).to(device)
        tokenizer = TokenizerCVTransformer(args.model)
    
    dataloader_train_val_test = {
        "train": create_dataloader(
            datas_to_use=datas_dict["train"],
            tokenizer=tokenizer,
            batch_size = args.batch_size,
            max_length = args.max_length
        ),
        "validation": create_dataloader(
            datas_to_use=datas_dict["validation"],
            tokenizer=tokenizer,
            batch_size = args.batch_size,
            max_length = args.max_length
        ),
        "test": create_dataloader(
            datas_to_use=datas_dict["test"],
            tokenizer=tokenizer,
            batch_size = args.batch_size,
            max_length = args.max_length
        ),
    }

    training_loop(
        model,
        config,
        dataloader_train_val_test,
        optimizer,
        scheduler
    )

################################################################################
# End of File
################################################################################