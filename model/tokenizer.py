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
    
    def __call__(self, input_vectors, max_length=512):
        return self.tokenizer(input_vectors, padding=True, truncation=True, return_tensors="pt", max_length=max_length)

################################################################################
# End of File
################################################################################