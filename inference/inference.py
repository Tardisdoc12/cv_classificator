################################################################################
# filename: inference.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 31/07,2025
################################################################################

import torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from model.model import ClassificatorCVTransformer
from model.tokenizer import TokenizerCVTransformer
from model.dataset_creater import get_text_from_cv

################################################################################

def main():
    args = get_arg_parser()
    model = ClassificatorCVTransformer(args.model).to(args.device)
    tokenizer = TokenizerCVTransformer(args.model)
    if not args.model_path is None:
        model.load_state_dict(torch.load(args.model_path))

    string_to_vectorize = f"Tag: {args.tag} CV: {get_text_from_cv(args.cv_path)}"
    print(string_to_vectorize)
    vectorized = tokenizer(string_to_vectorize, max_length=512).to(args.device)
    with torch.no_grad():
        output = model(vectorized["input_ids"], vectorized["attention_mask"])
    print("le score est:", output.argmax().item())

################################################################################

def get_arg_parser() -> Namespace:
    parser = ArgumentParser(description="Inference", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="prajjwal1/bert-tiny", help="Type of model to use")
    parser.add_argument("--model_path", type=str, help="Path of the model to load")
    parser.add_argument("--tag", type=str, help="Tag of the CV to use")
    parser.add_argument("--cv_path", type=str, help="Path of the CV to use")
    parser.add_argument("--device", choices=["cpu", "cuda"],type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
# End of File
################################################################################