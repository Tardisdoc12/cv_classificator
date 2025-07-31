################################################################################
# filename: tokenizer.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 31/07,2025
################################################################################

from transformers import AutoTokenizer

################################################################################

class TokenizerCVTransformer:
    def __init__(self,type = "prajjwal1/bert-tiny"):
        self.tokenizer = AutoTokenizer.from_pretrained(type)
    
    def __call__(self, input_vectors):
        return self.tokenizer(input_vectors, padding=True, truncation=True, return_tensors="pt")

################################################################################
# End of File
################################################################################