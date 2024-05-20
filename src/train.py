from transformers import AutoModel, AutoTokenizer
import torch
import random
import numpy as np
from model import AdaptedViT
import argparse

import os

def setup_model(pretrained_model):
    adapted_ViT = AdaptedViT()

    pretrained_state_dict = pretrained_model.model.image_encoder.state_dict()
    adapted_state_dict = adapted_ViT.state_dict()

    to_update = {module : weights for module, weights in pretrained_state_dict.items() if module in adapted_state_dict.keys() and adapted_state_dict[module].size() == weights.size()}

    print(f'Modules that are being updated : {to_update.keys()}')

    # Copy weights to our model
    adapted_state_dict.update(to_update)
    adapted_ViT.load_state_dict(adapted_state_dict)

    # Switch the image_encoders
    pretrained_model.image_encoder = adapted_ViT

    # Set all weights to frozen
    for param in pretrained_model.parameters():
        param.requires_grad = False 
    
    # Set new patch_embedding to trainable
    for param in pretrained_model.model.image_encoder.patch_embedding_equiv.parameters():
        param.requires_grad = True 
    
    # Set adapter module to trainable
    for param in pretrained_model.model.image_encoder.adapter.parameters():
        param.requires_grad = True 
    
    return pretrained_model
    

def main(args):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup model
    clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
    model = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=False)
    model.model.text_encoder.tokenizer = clip_tokenizer

    model = setup_model(model)

    model.train()
    model.to(device)




if __name__ == "__main__":
    main()