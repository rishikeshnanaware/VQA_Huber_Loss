You said:
#infer_from_videos.py

import os
import time
import argparse
import pickle as pkl
import random

import open_clip
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kendalltau as kendallr
from tqdm import tqdm

def encode_text_prompts(prompts, device="cuda"):
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        embedding = model.token_embedding(text_tokens)
        text_features = model.encode_text(text_tokens).float()
    return text_tokens, embedding, text_features

# Install DOVER
from dover import datasets
from dover import DOVER

import wandb

from model import TextEncoder, MaxVQA, EnhancedVisualEncoder

device = "cuda"

# Initialize datasets (only load val-maxwell)
with open("/home/ubuntu/MaxVQA/maxvqa.yml", "r") as f:
    opt = yaml.safe_load(f)

# Only use val-maxwell dataset
val_datasets = {}
val_datasets["val-maxwell"] = getattr(datasets, opt["data"]["val-maxwell"]["type"])(
    opt["data"]["val-maxwell"]["args"])

# Initialize clip
print(open_clip.list_pretrained())
model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
model = model.to(device)

# Initialize fast-vqa encoder
fast_vqa_encoder = DOVER(**opt["model"]["args"]).to(device)
fast_vqa_encoder.load_state_dict(torch.load("/home/ubuntu/MaxVQA/DOVER/pretrained_weights/DOVER.pth"), strict=False)

# Encode initialized prompts 
context = "X"

positive_descs = ["high quality", "good content", "organized composition", "vibrant color", "contrastive lighting", "consistent trajectory",
                  "good aesthetics", "sharp", "in-focus", "noiseless", "clear-motion", "stable", "well-exposed", 
                  "original", "fluent", "clear"]

negative_descs = ["low quality", "bad content", "chaotic composition", "faded color", "gloomy lighting", "incoherent trajectory",
                  "bad aesthetics", "fuzzy", "out-of-focus", "noisy", "blurry-motion", "shaky", "poorly-exposed",
                  "compressed", "choppy", "severely degraded"]

pos_prompts = [f"a {context} {desc} photo" for desc in positive_descs]
neg_prompts = [f"a {context} {desc} photo" for desc in negative_descs]

tokenizer = open_clip.get_tokenizer("RN50")
text_tokens, embedding, text_feats = encode_text_prompts(pos_prompts + neg_prompts, device=device)

# Load model
text_encoder = TextEncoder(model)
visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder)
maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).cuda()

state_dict = torch.load("/home/ubuntu/MaxVQA/maxvqa_maxwell.pt")
maxvqa.load_state_dict(state_dict)
maxvqa.initialize_inference(text_encoder)

# Evaluation
gts, paths = {}, {}

# Extract ground truth labels and paths
for val_name, val_dataset in val_datasets.items():
    gts[val_name] = [val_dataset.video_infos[i]["label"] for i in range(len(val_dataset))]

for val_name, val_dataset in val_datasets.items():
    paths[val_name] = [val_dataset.video_infos[i]["filename"] for i in range(len(val_dataset))]

val_prs = {}

# Iterate over datasets and evaluate
for val_name, val_dataset in val_datasets.items():
    if "train" in val_name:
        continue
    val_prs[val_name] = []
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=6, pin_memory=True,
    )
    
    print("Debugging")
    # Add this before the evaluation loop to check if the dataset is correctly initialized.
    print(f"Length of dataset {val_name}: {len(val_dataset)}")  # Check if the dataset has data

    # You can also print out some sample data from the dataset.
    if len(val_dataset) > 0:
        sample_data = val_dataset[0]  # Retrieve a single sample
        print(f"Sample data from the first entry: {sample_data}")  # Print a sample from the dataset


    for i, data in enumerate(tqdm(val_loader, desc=f"Evaluating in dataset [{val_name}].")):
        print(f"Data keys: {data.keys()}")  # Check if the expected keys are there
        print(f"Data sample {i} - Aesthetic shape: {data['aesthetic'].shape}, Technical shape: {data['technical'].shape}")  # Check if 'aesthetic' and 'technical' are loaded properly
        
        with torch.no_grad():
            print(f"Entering with torch.no_grad(): Iteration {i}")
            vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
            res = maxvqa(vis_feats, text_encoder, train=False)
            val_prs[val_name].extend(list(res.cpu().numpy()))
            print(f"Predictions for batch {i}: {val_prs[val_name][-1]}")
        torch.cuda.empty_cache()

    print("Debugging")

    # Align predictions and ground truth labels
    val_gts = gts[val_name]
    if val_name != "val-maxwell":
        for i in range(16):
            # Ensure both predictions and ground truths have the same length
            predictions = [pr[i] for pr in val_prs[val_name]]
            ground_truths = val_gts
            min_len = min(len(predictions), len(ground_truths))
            predictions = predictions[:min_len]
            ground_truths = ground_truths[:min_len]
            
            if len(predictions) >= 2:  # Ensure at least 2 elements for Pearson correlation
                print(f"Generalization Evaluating: {positive_descs[i]}<->{negative_descs[i]}", pearsonr(predictions, ground_truths)[0])
            else:
                print(f"Skipping evaluation for {positive_descs[i]}<->{negative_descs[i]}: Not enough data points.")
    else:
        import pandas as pd
        max_gts = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_val.csv")
        for key, i in zip(max_gts, range(16)):
            predictions = [pr[i] for pr in val_prs[val_name]]
            ground_truths = max_gts[key].tolist()
            
            # Ensure both predictions and ground truths have the same length
            min_len = min(len(predictions), len(ground_truths))
            predictions = predictions[:min_len]
            ground_truths = ground_truths[:min_len]
            
            if len(predictions) >= 2:  # Ensure at least 2 elements for Pearson correlation
                print(f"Evaluating {key}: {positive_descs[i]}<->{negative_descs[i]}", pearsonr(predictions, ground_truths)[0])
            else:
                print(f"Skipping evaluation for {key}: Not enough data points.")

# Save the results
with open("maxvqa_global_results.pkl", "wb") as f:
    pkl.dump(val_prs, f)