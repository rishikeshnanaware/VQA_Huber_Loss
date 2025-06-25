''' Working code
import os
import glob
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
import time
import pandas as pd

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float() #+ 0.3 * rank_loss(y_pred[...,None], y[...,None])

def count_parameters(model):
    for name, module in model.named_children():
        print(name, "|", sum(p.numel() for p in module.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MaxVisualFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, visual_features, max_gts, indices=None):
        super().__init__()
        if indices is None:
            indices = range(len(visual_features))
        self.visual_features = [visual_features[ind] for ind in indices]
        self.gts = [max_gts.iloc[ind].values for ind in indices]

    def __getitem__(self, index):
        return self.visual_features[index], torch.Tensor(self.gts[index])

    def __len__(self):
        return len(self.gts)

from dover import datasets
from dover import DOVER

import wandb

from model import TextEncoder, MaxVQA, EnhancedVisualEncoder

device = "cuda"
with open("maxvqa.yml", "r") as f:
    opt = yaml.safe_load(f)

val_datasets = {}
for name, dataset in opt["data"].items():
    val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

print(open_clip.list_pretrained())
model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
model = model.to(device)

fast_vqa_encoder = DOVER(**opt["model"]["args"]).to(device)
fast_vqa_encoder.load_state_dict(torch.load("/home/ubuntu/MaxVQA/DOVER/pretrained_weights/DOVER.pth"), strict=False)

context = "X"

positive_descs = ["high quality", "good content", "organized composition", "vibrant color", "contrastive lighting", "consistent trajectory",
                  "good aesthetics", "sharp", "in-focus", "noiseless", "clear-motion", "stable", "well-exposed", 
                  "original", "fluent", "clear"]

negative_descs = ["low quality", "bad content", "chaotic composition", "faded color", "gloomy lighting", "incoherent trajectory",
                  "bad aesthetics", "fuzzy", "out-of-focus", "noisy", "blurry-motion", "shaky", "poorly-exposed",
                  "compressed", "choppy", "severely degraded"]

pos_prompts = [ f"a {context} {desc} photo" for desc in positive_descs]
neg_prompts = [ f"a {context} {desc} photo" for desc in negative_descs]

tokenizer = open_clip.get_tokenizer("RN50")

def encode_text_prompts(prompts, device="cuda"):
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        embedding = model.token_embedding(text_tokens)
        text_features = model.encode_text(text_tokens).float()
    return text_tokens, embedding, text_features

text_tokens, embedding, text_feats = encode_text_prompts(pos_prompts + neg_prompts, device=device)

text_encoder = TextEncoder(model).to(device)
visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder).to(device)

maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

gts, paths = {}, {}

for val_name, val_dataset in val_datasets.items():
    gts[val_name] = [val_dataset.video_infos[i]["label"] for i in range(len(val_dataset))]

for val_name, val_dataset in val_datasets.items():
    paths[val_name] = [val_dataset.video_infos[i]["filename"] for i in range(len(val_dataset))]

feats = {}

os.makedirs("features", exist_ok=True)

# Extract features based on first 1/4th of the dataset (for training and validation sets)
for val_name, val_dataset in val_datasets.items():
    if "maxwell" not in val_name:
        print(f"Omitting {val_name}")
        continue
    feat_path = f"features/maxvqa_vis_{val_name}.pkl"
    if glob.glob(feat_path):
        print("Found pre-extracted visual features...")
        s = time.time()
        feats[val_name] = torch.load(feat_path)
        print(f"Successfully loaded {val_name}, elapsed {time.time() - s:.2f}s.")
    else:
        print("Extracting on-the-fly...")
        feats[val_name] = []
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=8, pin_memory=True,
        )
        for i, data in enumerate(tqdm(val_loader, desc=f"Extracting in dataset [{val_name}].")):
            if val_name == "train-maxwell" and i >= len(val_dataset) // 4:  # Only extract the first 1/4th for train
                break
            if val_name == "val-maxwell" and i >= len(val_dataset) // 4:  # Only extract the first 1/4th for validation
                break
            with torch.no_grad():
                vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
                feats[val_name].append(vis_feats.half().cpu())
            torch.cuda.empty_cache()
        torch.save(feats[val_name], feat_path)

max_gts_train = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_train.csv")
max_gts_val = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_val.csv")

# Select the first 1/4th of the data
train_subset_indices = list(range(len(max_gts_train) // 4))
val_subset_indices = list(range(len(max_gts_val) // 4))

train_dataset = MaxVisualFeatureDataset(feats["train-maxwell"], max_gts_train, train_subset_indices)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = MaxVisualFeatureDataset(feats["val-maxwell"], max_gts_val, val_subset_indices)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

print(f'The model has {count_parameters(maxvqa):,} trainable parameters')
optimizer = torch.optim.AdamW(maxvqa.parameters(), lr=1e-3)

maxvqa_ema = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

if True:
    val_prs, val_gts = [], []
    for data in tqdm(test_dataloader):
        with torch.no_grad():
            vis_feat, gt = data
            res = maxvqa(vis_feat.cuda(), text_encoder)[:, 0]
            val_prs.extend(list(res.cpu().numpy()))
            val_gts.extend(list(gt.cpu().numpy()))
    val_prs = np.stack(val_prs, 0)
    val_gts = np.stack(val_gts, 0)

    for i, key in zip(range(16), max_gts_train):
        srcc, plcc = spearmanr(val_prs[:, i], val_gts[:, i])[0], pearsonr(val_prs[:, i], val_gts[:, i])[0]
        print(key, srcc, plcc)

best_all_plcc = 0

run = wandb.init(
    project="MaxVQA",
    name=f"maxvqa_maxwell_pushed",
    reinit=True,
    settings=wandb.Settings(start_method="thread"),
)

for epoch in range(30):
    print(epoch)
    maxvqa.train()
    for data in tqdm(train_dataloader):
        optimizer.zero_grad()
        vis_feat, gt = data
        res = maxvqa(vis_feat.cuda(), text_encoder)
        loss, aux_loss = 0, 0
        for i in range(16):
            loss += plcc_loss(res[:, 0, i], gt[:, i].cuda().float())
            for j in range(i + 1, 16):
                aux_loss += 0.005 * (0.5 - plcc_loss(res[:, 0, i], res[0, :, j]))

        wandb.log({"loss": loss.item(), "aux_loss": aux_loss.item()})
        loss += aux_loss
        loss.backward()
        optimizer.step()

        model_params = dict(maxvqa.named_parameters())
        model_ema_params = dict(maxvqa_ema.named_parameters())
        for k in model_params.keys():
            model_ema_params[k].data.mul_(0.999).add_(
                model_params[k].data, alpha=1 - 0.999)

    maxvqa.eval()

    val_prs, val_gts = [], []
    for data in tqdm(test_dataloader):
        with torch.no_grad():
            vis_feat, gt = data
            res = maxvqa_ema(vis_feat.cuda(), text_encoder)[:, 0]
            val_prs.extend(list(res.cpu().numpy()))
            val_gts.extend(list(gt.cpu().numpy()))
    val_prs = np.stack(val_prs, 0)
    val_gts = np.stack(val_gts, 0)

    all_plcc = 0
    for i, key in zip(range(16), max_gts_train):
        srcc, plcc = spearmanr(val_prs[:, i], val_gts[:, i])[0], pearsonr(val_prs[:, i], val_gts[:, i])[0]
        print(key, srcc, plcc)
        all_plcc += plcc

    if all_plcc > best_all_plcc:
        with open("maxvqa_validation_results.pkl", "wb") as f:
            pkl.dump(val_prs, f)
        best_all_plcc = all_plcc
        torch.save(maxvqa_ema.state_dict(), "maxvqa_pushed_away_maxwell.pt")
'''
#Experiment 1 Training on first 912 samples
# import os
# import glob
# import argparse
# import pickle as pkl
# import random
# import open_clip
# import numpy as np
# import torch
# import torch.nn as nn
# import yaml
# from scipy.stats import pearsonr, spearmanr
# from scipy.stats import kendalltau as kendallr
# from tqdm import tqdm
# import time
# import pandas as pd

# # Define your loss functions
# def rank_loss(y_pred, y):
#     ranking_loss = torch.nn.functional.relu(
#         (y_pred - y_pred.t()) * torch.sign((y.t() - y))
#     )
#     scale = 1 + torch.max(ranking_loss)
#     return (
#         torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
#     ).float()

# def plcc_loss(y_pred, y):
#     sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
#     y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
#     sigma, m = torch.std_mean(y, unbiased=False)
#     y = (y - m) / (sigma + 1e-8)
#     loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
#     rho = torch.mean(y_pred * y)
#     loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
#     return ((loss0 + loss1) / 2).float()  #+ 0.3 * rank_loss(y_pred[...,None], y[...,None])

# def count_parameters(model):
#     for name, module in model.named_children():
#         print(name, "|", sum(p.numel() for p in module.parameters() if p.requires_grad))
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class MaxVisualFeatureDataset(torch.utils.data.Dataset):
#     def __init__(self, visual_features, max_gts, indices=None):
#         super().__init__()
#         if indices == None:
#             indices = range(len(visual_features))
#             print("Using all indices:", indices)
#         self.visual_features = [visual_features[ind] for ind in indices]
#         self.gts = [max_gts.iloc[ind].values for ind in indices]

#     def __getitem__(self, index):
#         return self.visual_features[index], torch.Tensor(self.gts[index])

#     def __len__(self):
#         return len(self.gts)

# def encode_text_prompts(prompts, device="cuda"):
#     text_tokens = tokenizer(prompts).to(device)
#     with torch.no_grad():
#         embedding = model.token_embedding(text_tokens)
#         text_features = model.encode_text(text_tokens).float()
#     return text_tokens, embedding, text_features

# # You need to install DOVER
# from dover import datasets
# from dover import DOVER
# import wandb
# from model import TextEncoder, MaxVQA, EnhancedVisualEncoder

# device = "cuda"

# # Initialize datasets
# with open("maxvqa.yml", "r") as f:
#     opt = yaml.safe_load(f)

# val_datasets = {}
# for name, dataset in opt["data"].items():
#     val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

# # Initialize clip
# print(open_clip.list_pretrained())
# model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
# model = model.to(device)

# # Initialize fast-vqa encoder
# fast_vqa_encoder = DOVER(**opt["model"]["args"]).to(device)
# fast_vqa_encoder.load_state_dict(torch.load("/home/ubuntu/MaxVQA/DOVER/pretrained_weights/DOVER.pth"), strict=False)

# # Encode initialized prompts
# context = "X"

# positive_descs = ["high quality", "good content", "organized composition", "vibrant color", "contrastive lighting", "consistent trajectory",
#                   "good aesthetics", "sharp", "in-focus", "noiseless", "clear-motion", "stable", "well-exposed",
#                   "original", "fluent", "clear"]

# negative_descs = ["low quality", "bad content", "chaotic composition", "faded color", "gloomy lighting", "incoherent trajectory",
#                   "bad aesthetics", "fuzzy", "out-of-focus", "noisy", "blurry-motion", "shaky", "poorly-exposed",
#                   "compressed", "choppy", "severely degraded"]

# pos_prompts = [f"a {context} {desc} photo" for desc in positive_descs]
# neg_prompts = [f"a {context} {desc} photo" for desc in negative_descs]

# tokenizer = open_clip.get_tokenizer("RN50")
# text_tokens, embedding, text_feats = encode_text_prompts(pos_prompts + neg_prompts, device=device)

# # Load model
# text_encoder = TextEncoder(model).to(device)
# visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder).to(device)
# maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# # Extract Features before training
# gts, paths = {}, {}

# # Read the ground truth values from CSV file
# max_gts_train = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_train.csv")
# max_gts_train = max_gts_train.iloc[:912]
# max_gts_val = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_val.csv")  # Assuming separate validation set CSV

# # Make sure features are loaded from pre-saved files if they exist, or extract them
# feats = {}
# print("Loading pre-extracted features...")

# # Assuming you already have a feature extraction process like this:
# for val_name, val_dataset in val_datasets.items():
#     if "maxwell" not in val_name:
#         print(f"Omitting {val_name}")
#         continue
#     feat_path = f"features/maxvqa_vis_{val_name}.pkl"
#     if glob.glob(feat_path):
#         print(f"Found pre-extracted visual features for {val_name}...")
#         s = time.time()
#         feats[val_name] = torch.load(feat_path)
#         print(f"Successfully loaded {val_name}, elapsed {time.time() - s:.2f}s.")
#     else:
#         print(f"Extracting features for {val_name}...")
#         feats[val_name] = []
#         val_loader = torch.utils.data.DataLoader(
#             val_dataset, batch_size=1, num_workers=8, pin_memory=True,
#         )
#         for i, data in enumerate(tqdm(val_loader, desc=f"Extracting in dataset [{val_name}].")):
#             if val_name == "train-maxwell" and i >= 912:  # Only extract the first 1/4th for train
#                 break
#             if val_name == "val-maxwell" and i >= 909:  # Only extract the first 1/4th for train
#                 break
#             with torch.no_grad():
#                 vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
#                 feats[val_name].append(vis_feats.half().cpu())
#             torch.cuda.empty_cache()
#         torch.save(feats[val_name], feat_path)

# # Ensure the model parameters are counted
# print(f'The model has {count_parameters(maxvqa):,} trainable parameters')
# optimizer = torch.optim.AdamW(maxvqa.parameters(), lr=1e-3)

# # Create the train and test datasets
# train_dataset = MaxVisualFeatureDataset(feats["train-maxwell"], max_gts_train)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# test_dataset = MaxVisualFeatureDataset(feats["val-maxwell"], max_gts_val)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

# maxvqa_ema = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# # Initialize WandB
# run = wandb.init(
#     project="MaxVQA",
#     name=f"maxvqa_maxwell_pushed",
#     reinit=True,
#     settings=wandb.Settings(start_method="thread"),
# )

# # Training loop
# best_all_plcc = 0

# for epoch in range(20):
#     print(epoch)
#     maxvqa.train()
#     for data in tqdm(train_dataloader):
#         optimizer.zero_grad()
#         vis_feat, gt = data
#         res = maxvqa(vis_feat.cuda(), text_encoder)
#         loss, aux_loss = 0, 0
#         for i in range(16):
#             loss += plcc_loss(res[:, 0, i], gt[:, i].cuda().float())
#             for j in range(i + 1, 16):
#                 aux_loss += 0.005 * (0.5 - plcc_loss(res[:, 0, i], res[0, :, j]))
#         wandb.log({"loss": loss.item(), "aux_loss": aux_loss.item()})
#         loss += aux_loss
#         loss.backward()
#         optimizer.step()

#         model_params = dict(maxvqa.named_parameters())
#         model_ema_params = dict(maxvqa_ema.named_parameters())
#         for k in model_params.keys():
#             model_ema_params[k].data.mul_(0.999).add_(
#                 model_params[k].data, alpha=1 - 0.999)

#     # Evaluation phase
#     maxvqa.eval()
#     val_prs, val_gts = [], []
#     for data in tqdm(test_dataloader):
#         with torch.no_grad():
#             vis_feat, gt = data
#             res = maxvqa_ema(vis_feat.cuda(), text_encoder)[:, 0]
#             val_prs.extend(list(res.cpu().numpy()))
#             val_gts.extend(list(gt.cpu().numpy()))

#     val_prs = np.stack(val_prs, 0)
#     val_gts = np.stack(val_gts, 0)

#     all_plcc = 0
#     for i, key in zip(range(16), max_gts_train):
#         srcc, plcc = spearmanr(val_prs[:, i], val_gts[:, i])[0], pearsonr(val_prs[:, i], val_gts[:, i])[0]
#         print(key, srcc, plcc)
#         all_plcc += plcc

#     if all_plcc > best_all_plcc:
#         with open("maxvqa_validation_results.pkl", "wb") as f:
#             pkl.dump(val_prs, f)
#         best_all_plcc = all_plcc
#         torch.save(maxvqa_ema.state_dict(), "maxvqa_pushed_away_maxwell.pt")

'''#Training model chuck by chunk
import os
import glob
import pickle as pkl
import random
import open_clip
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import time
import pandas as pd

# Define your loss functions
def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def count_parameters(model):
    for name, module in model.named_children():
        print(name, "|", sum(p.numel() for p in module.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MaxVisualFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, visual_features, max_gts, indices=None):
        super().__init__()
        if indices is None:
            indices = range(len(visual_features))
            print("Using all indices:", indices)
        self.visual_features = [visual_features[ind] for ind in indices]
        self.gts = [max_gts.iloc[ind].values for ind in indices]

    def __getitem__(self, index):
        return self.visual_features[index], torch.Tensor(self.gts[index])

    def __len__(self):
        return len(self.gts)

def encode_text_prompts(prompts, device="cuda"):
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        embedding = model.token_embedding(text_tokens)
        text_features = model.encode_text(text_tokens).float()
    return text_tokens, embedding, text_features

# You need to install DOVER
from dover import datasets
from dover import DOVER
import wandb
from model import TextEncoder, MaxVQA, EnhancedVisualEncoder

device = "cuda"

# Initialize datasets
with open("maxvqa.yml", "r") as f:
    opt = yaml.safe_load(f)

val_datasets = {}
for name, dataset in opt["data"].items():
    val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

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
text_encoder = TextEncoder(model).to(device)
visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder).to(device)
maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# Read the ground truth values from CSV file
max_gts_train = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_train.csv")
max_gts_val = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_val.csv")  # Assuming separate validation set CSV

# Ensure the model parameters are counted
print(f'The model has {count_parameters(maxvqa):,} trainable parameters')
optimizer = torch.optim.AdamW(maxvqa.parameters(), lr=1e-3)

# Make sure this directory exists to store extracted features
os.makedirs("features", exist_ok=True)

# Path for validation features
val_feat_path = "features/maxvqa_vis_val-maxwell.pkl"

# Extract or load validation features
if glob.glob(val_feat_path):
    print("Found pre-extracted visual features for validation set...")
    s = time.time()
    val_feats = torch.load(val_feat_path)
    print(f"Successfully loaded validation set features, elapsed {time.time() - s:.2f}s.")
else:
    print("Extracting features for validation set...")
    val_feats = []
    val_loader = torch.utils.data.DataLoader(
        val_datasets["val-maxwell"], batch_size=1, num_workers=8, pin_memory=True
    )
    for i, data in enumerate(tqdm(val_loader, desc="Extracting features for validation set")):
        if i >= 909:  # Limit to first 909 samples
            break
        with torch.no_grad():
            vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
            val_feats.append(vis_feats.half().cpu())
        torch.cuda.empty_cache()
    
    torch.save(val_feats, val_feat_path)
    print(f"Validation features saved to {val_feat_path}")

# Create the validation dataset and dataloader
test_dataset = MaxVisualFeatureDataset(val_feats, max_gts_val)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

maxvqa_ema = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# Initialize WandB
run = wandb.init(
    project="MaxVQA",
    name=f"maxvqa_maxwell_pushed",
    reinit=True,
    settings=wandb.Settings(start_method="thread"),
)

# Define chunks for training data
chunk_sizes = [912, 912, 912, 896]
num_chunks = len(chunk_sizes)
best_all_plcc = 0

train_losses = []
val_losses = []

for chunk_num in range(num_chunks):
    s = chunk_num * 912
    e = s + chunk_sizes[chunk_num]
    indices = range(s, e)
    print(f"Processing chunk {chunk_num + 1}/{num_chunks}: {s}-{e}")

    feat_path = f"features/maxvqa_vis_train_chunk_{chunk_num + 1}.pkl"
    if glob.glob(feat_path):
        print(f"Found pre-extracted visual features for chunk {chunk_num + 1}...")
        s = time.time()
        feats_chunk = torch.load(feat_path)
        print(f"Successfully loaded chunk {chunk_num + 1}, elapsed {time.time() - s:.2f}s.")
    else:
        print(f"Extracting features for training chunk {chunk_num + 1}...")
        feats_chunk = []
        for i in tqdm(indices, desc=f"Extracting features [{s}-{e}]"):
            data = val_datasets["train-maxwell"][i]  # Assuming "train-maxwell" is the dataset key
            with torch.no_grad():
                vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
                feats_chunk.append(vis_feats.half().cpu())
            torch.cuda.empty_cache()
        
        torch.save(feats_chunk, feat_path)
        print(f"Features for chunk {chunk_num + 1} saved to {feat_path}")

    # Create the train dataset and dataloader for this chunk
    train_dataset_chunk = MaxVisualFeatureDataset(feats_chunk, max_gts_train.iloc[indices])
    train_dataloader_chunk = torch.utils.data.DataLoader(train_dataset_chunk, batch_size=16, shuffle=True)

    # Training loop for this chunk
    maxvqa.train()
    for epoch in range(20):
        print(f"Epoch {epoch + 1}/20")
        epoch_loss = 0
        for data in tqdm(train_dataloader_chunk):
            optimizer.zero_grad()
            vis_feat, gt = data
            res = maxvqa(vis_feat.cuda(), text_encoder)
            loss, aux_loss = 0, 0
            for i in range(16):
                loss += plcc_loss(res[:, 0, i], gt[:, i].cuda().float())
                for j in range(i + 1, 16):
                    aux_loss += 0.005 * (0.5 - plcc_loss(res[:, 0, i], res[0, :, j]))
            wandb.log({"loss": loss.item(), "aux_loss": aux_loss.item()})
            loss += aux_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            model_params = dict(maxvqa.named_parameters())
            model_ema_params = dict(maxvqa_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999)

        # Append average training loss for the epoch
        train_losses.append(epoch_loss / len(train_dataloader_chunk))

        # Evaluation phase
        maxvqa.eval()
        val_loss = 0        
        val_prs, val_gts = [], []
        for data in tqdm(test_dataloader):
            with torch.no_grad():
                vis_feat, gt = data
                res = maxvqa_ema(vis_feat.cuda(), text_encoder)[:, 0]
                val_prs.extend(list(res.cpu().numpy()))
                val_gts.extend(list(gt.cpu().numpy()))
                # Calculate validation loss
                val_loss += plcc_loss(res, gt.cuda().float()).item()
        val_losses.append(val_loss / len(test_dataloader))

        val_prs = np.stack(val_prs, 0)
        val_gts = np.stack(val_gts, 0)

        all_plcc = 0
        for i, key in zip(range(16), max_gts_train):
            srcc, plcc = spearmanr(val_prs[:, i], val_gts[:, i])[0], pearsonr(val_prs[:, i], val_gts[:, i])[0]
            print(key, srcc, plcc)
            all_plcc += plcc

        if all_plcc > best_all_plcc:
            with open("maxvqa_validation_results.pkl", "wb") as f:
                pkl.dump(val_prs, f)
            best_all_plcc = all_plcc
            torch.save(maxvqa_ema.state_dict(), f"maxvqa_best_chunk_{chunk_num + 1}.pt")


# Plot learning curves after training
import matplotlib.pyplot as plt
print("Train Losses: ", len(train_losses))
print("Val Losses: ", len(val_losses))
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves (Loss)')
plt.savefig('results.png')
plt.show()
'''

# #Implemented Early stopping, learning rate scheduling and L2 regularization.
# import os
# import glob
# import pickle as pkl
# import random
# import open_clip
# import numpy as np
# import torch
# import torch.nn as nn
# import yaml
# from scipy.stats import pearsonr, spearmanr
# from tqdm import tqdm
# import time
# import pandas as pd

# # Define your loss functions
# def rank_loss(y_pred, y):
#     ranking_loss = torch.nn.functional.relu(
#         (y_pred - y_pred.t()) * torch.sign((y.t() - y))
#     )
#     scale = 1 + torch.max(ranking_loss)
#     return (
#         torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
#     ).float()

# def plcc_loss(y_pred, y):
#     sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
#     y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
#     sigma, m = torch.std_mean(y, unbiased=False)
#     y = (y - m) / (sigma + 1e-8)
#     loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
#     rho = torch.mean(y_pred * y)
#     loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
#     return ((loss0 + loss1) / 2).float()

# def count_parameters(model):
#     for name, module in model.named_children():
#         print(name, "|", sum(p.numel() for p in module.parameters() if p.requires_grad))
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class MaxVisualFeatureDataset(torch.utils.data.Dataset):
#     def __init__(self, visual_features, max_gts, indices=None):
#         super().__init__()
#         if indices is None:
#             indices = range(len(visual_features))
#             print("Using all indices:", indices)
#         self.visual_features = [visual_features[ind] for ind in indices]
#         self.gts = [max_gts.iloc[ind].values for ind in indices]

#     def __getitem__(self, index):
#         return self.visual_features[index], torch.Tensor(self.gts[index])

#     def __len__(self):
#         return len(self.gts)

# def encode_text_prompts(prompts, device="cuda"):
#     text_tokens = tokenizer(prompts).to(device)
#     with torch.no_grad():
#         embedding = model.token_embedding(text_tokens)
#         text_features = model.encode_text(text_tokens).float()
#     return text_tokens, embedding, text_features

# # You need to install DOVER
# from dover import datasets
# from dover import DOVER
# import wandb
# from model import TextEncoder, MaxVQA, EnhancedVisualEncoder

# device = "cuda"

# # Initialize datasets
# with open("maxvqa.yml", "r") as f:
#     opt = yaml.safe_load(f)

# val_datasets = {}
# for name, dataset in opt["data"].items():
#     val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

# # Initialize clip
# print(open_clip.list_pretrained())
# model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
# model = model.to(device)

# # Initialize fast-vqa encoder
# fast_vqa_encoder = DOVER(**opt["model"]["args"]).to(device)
# fast_vqa_encoder.load_state_dict(torch.load("/home/ubuntu/MaxVQA/DOVER/pretrained_weights/DOVER.pth"), strict=False)

# # Encode initialized prompts
# context = "X"

# positive_descs = ["high quality", "good content", "organized composition", "vibrant color", "contrastive lighting", "consistent trajectory",
#                   "good aesthetics", "sharp", "in-focus", "noiseless", "clear-motion", "stable", "well-exposed",
#                   "original", "fluent", "clear"]

# negative_descs = ["low quality", "bad content", "chaotic composition", "faded color", "gloomy lighting", "incoherent trajectory",
#                   "bad aesthetics", "fuzzy", "out-of-focus", "noisy", "blurry-motion", "shaky", "poorly-exposed",
#                   "compressed", "choppy", "severely degraded"]

# pos_prompts = [f"a {context} {desc} photo" for desc in positive_descs]
# neg_prompts = [f"a {context} {desc} photo" for desc in negative_descs]

# tokenizer = open_clip.get_tokenizer("RN50")
# text_tokens, embedding, text_feats = encode_text_prompts(pos_prompts + neg_prompts, device=device)

# # Load model
# text_encoder = TextEncoder(model).to(device)
# visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder).to(device)
# maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# # Read the ground truth values from CSV file
# max_gts_train = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_train.csv")
# max_gts_val = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_val.csv").iloc[:909]  # Limit to first 909 samples

# # Ensure the model parameters are counted
# print(f'The model has {count_parameters(maxvqa):,} trainable parameters')

# # Initialize optimizer with weight decay (L2 regularization)
# optimizer = torch.optim.AdamW(maxvqa.parameters(), lr=1e-3, weight_decay=1e-4)

# # Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# # Early stopping configuration
# early_stopping_patience = 5
# no_improve_epochs = 0
# best_all_plcc = 0

# # Make sure this directory exists to store extracted features
# os.makedirs("features", exist_ok=True)

# # Path for validation features
# val_feat_path = "features/maxvqa_vis_val-maxwell.pkl"

# # Extract or load validation features
# if glob.glob(val_feat_path):
#     print("Found pre-extracted visual features for validation set...")
#     s = time.time()
#     val_feats = torch.load(val_feat_path)
#     print(f"Successfully loaded validation set features, elapsed {time.time() - s:.2f}s.")
# else:
#     print("Extracting features for validation set...")
#     val_feats = []
#     val_loader = torch.utils.data.DataLoader(
#         val_datasets["val-maxwell"], batch_size=1, num_workers=8, pin_memory=True
#     )
#     for i, data in enumerate(tqdm(val_loader, desc="Extracting features for validation set")):
#         if i >= 909:  # Limit to first 909 samples
#             break
#         with torch.no_grad():
#             vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
#             val_feats.append(vis_feats.half().cpu())
#         torch.cuda.empty_cache()
    
#     torch.save(val_feats, val_feat_path)
#     print(f"Validation features saved to {val_feat_path}")

# # Create the validation dataset and dataloader
# test_dataset = MaxVisualFeatureDataset(val_feats, max_gts_val)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

# maxvqa_ema = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# # Initialize WandB
# run = wandb.init(
#     project="MaxVQA",
#     name=f"maxvqa_maxwell_pushed",
#     reinit=True,
#     settings=wandb.Settings(start_method="thread"),
# )

# # Define chunks for training data
# chunk_sizes = [912, 912, 912, 896]
# num_chunks = len(chunk_sizes)

# train_losses = []
# val_losses = []


# for chunk_num in range(num_chunks):
#     s = chunk_num * 912
#     e = s + chunk_sizes[chunk_num]
#     indices = range(s, e)
#     print(f"Processing chunk {chunk_num + 1}/{num_chunks}: {s}-{e}")

#     feat_path = f"features/maxvqa_vis_train_chunk_{chunk_num + 1}.pkl"
#     if glob.glob(feat_path):
#         print(f"Found pre-extracted visual features for chunk {chunk_num + 1}...")
#         s = time.time()
#         feats_chunk = torch.load(feat_path)
#         print(f"Successfully loaded chunk {chunk_num + 1}, elapsed {time.time() - s:.2f}s.")
#     else:
#         print(f"Extracting features for training chunk {chunk_num + 1}...")
#         feats_chunk = []
#         for i in tqdm(indices, desc=f"Extracting features [{s}-{e}]"):
#             data = val_datasets["train-maxwell"][i]  # Assuming "train-maxwell" is the dataset key
#             with torch.no_grad():
#                 vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
#                 feats_chunk.append(vis_feats.half().cpu())
#             torch.cuda.empty_cache()
        
#         torch.save(feats_chunk, feat_path)
#         print(f"Features for chunk {chunk_num + 1} saved to {feat_path}")

#     # Create the train dataset and dataloader for this chunk
#     train_dataset_chunk = MaxVisualFeatureDataset(feats_chunk, max_gts_train.iloc[indices])
#     train_dataloader_chunk = torch.utils.data.DataLoader(train_dataset_chunk, batch_size=16, shuffle=True)

#     # Training loop for this chunk

#     maxvqa.train()
#     for epoch in range(20):
#         print(f"Epoch {epoch + 1}/20")
#         epoch_loss = 0
        
#         # Loop over training data
#         for data in tqdm(train_dataloader_chunk):
#             optimizer.zero_grad()
#             vis_feat, gt = data
#             res = maxvqa(vis_feat.cuda(), text_encoder)
#             loss, aux_loss = 0, 0
#             for i in range(16):
#                 loss += plcc_loss(res[:, 0, i], gt[:, i].cuda().float())
#                 for j in range(i + 1, 16):
#                     aux_loss += 0.005 * (0.5 - plcc_loss(res[:, 0, i], res[0, :, j]))
            
#             loss += aux_loss
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
            
#             wandb.log({"loss": loss.item(), "aux_loss": aux_loss.item()})
#             print(f"Batch loss: {loss.item()}")

#         # Append average training loss for the epoch
#         train_losses.append(epoch_loss / len(train_dataloader_chunk))

#         # Update the learning rate using the scheduler
#         scheduler.step()

#         # Print loss for the entire epoch
#         print(f"Epoch {epoch + 1} loss: {epoch_loss}")

#         # Evaluation phase for validation
#         import numpy as np

#         # Evaluation phase for validation
#         maxvqa.eval()
#         val_loss = 0
#         val_prs, val_gts = [], []
#         for data in tqdm(test_dataloader):
#             with torch.no_grad():
#                 vis_feat, gt = data
#                 res = maxvqa_ema(vis_feat.cuda(), text_encoder)[:, 0]
                
#                 # Resize res[:, 0] to match gt if necessary
#                 if res.size(1) != gt.size(1):
#                     res = res[:, :gt.size(1)]
                
#                 val_prs.extend(list(res.cpu().numpy()))
#                 val_gts.extend(list(gt.cpu().numpy()))
                
#                 # Calculate validation loss
#                 val_loss += plcc_loss(res, gt.cuda().float()).item()

#         val_losses.append(val_loss / len(test_dataloader))

#         # Convert lists to NumPy arrays
#         val_prs = np.array(val_prs)
#         val_gts = np.array(val_gts)

#         # Log validation loss
#         wandb.log({"val_loss": val_loss})

#         # Calculate and log metrics
#         all_plcc = 0
#         for i, key in zip(range(16), max_gts_train.columns):
#             srcc, plcc = spearmanr(val_prs[:, i], val_gts[:, i])[0], pearsonr(val_prs[:, i], val_gts[:, i])[0]
#             wandb.log({f"{key}_srcc": srcc, f"{key}_plcc": plcc})
#             print(key, srcc, plcc)
#             all_plcc += plcc
        
#         # Check for improvement and implement early stopping
#         # Existing early stopping code...
#         if all_plcc > best_all_plcc:
#             print("Saving model")
#             with open("maxvqa_validation_results.pkl", "wb") as f:
#                 pkl.dump(val_prs, f)
#             best_all_plcc = all_plcc
#             torch.save(maxvqa_ema.state_dict(), f"maxvqa_best_chunk_{chunk_num + 1}.pt")
#             no_improve_epochs = 0
#         else:
#             no_improve_epochs += 1
#             print(f"No improvement for {no_improve_epochs} epoch(s)")
            
#         # Early stopping condition
#         # if no_improve_epochs >= early_stopping_patience:
#         #     print("Early stopping triggered")
#         #     break


# # Plot learning curves after training
# import matplotlib.pyplot as plt
# print("Train Losses: ", len(train_losses))
# print("Val Losses: ", len(val_losses))
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
# plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Learning Curves (Loss)')
# plt.savefig('results.png')
# plt.show()


#     # maxvqa.train()
#     # for epoch in range(20):
#     #     print(f"Epoch {epoch + 1}/20")
#     #     epoch_loss = 0
        
#     #     for data in tqdm(train_dataloader_chunk):
#     #         optimizer.zero_grad()
#     #         vis_feat, gt = data
#     #         res = maxvqa(vis_feat.cuda(), text_encoder)
#     #         loss, aux_loss = 0, 0
#     #         for i in range(16):
#     #             loss += plcc_loss(res[:, 0, i], gt[:, i].cuda().float())
#     #             for j in range(i + 1, 16):
#     #                 aux_loss += 0.005 * (0.5 - plcc_loss(res[:, 0, i], res[0, :, j]))
            
#     #         loss += aux_loss
#     #         loss.backward()
#     #         optimizer.step()
            
#     #         epoch_loss += loss.item()
            
#     #         wandb.log({"loss": loss.item(), "aux_loss": aux_loss.item()})
            
#     #         # Print loss for the current batch
#     #         print(f"Batch loss: {loss.item()}")

#     #         model_params = dict(maxvqa.named_parameters())
#     #         model_ema_params = dict(maxvqa_ema.named_parameters())
#     #         for k in model_params.keys():
#     #             model_ema_params[k].data.mul_(0.999).add_(
#     #                 model_params[k].data, alpha=1 - 0.999)
        
#     #     # Update the learning rate using the scheduler
#     #     scheduler.step()

#     #     # Print loss for the entire epoch
#     #     print(f"Epoch {epoch + 1} loss: {epoch_loss}")

#     #     # Evaluation phase
#     #     maxvqa.eval()
#     #     val_prs, val_gts = [], []
#     #     for data in tqdm(test_dataloader):
#     #         with torch.no_grad():
#     #             vis_feat, gt = data
#     #             res = maxvqa_ema(vis_feat.cuda(), text_encoder)[:, 0]
#     #             val_prs.extend(list(res.cpu().numpy()))
#     #             val_gts.extend(list(gt.cpu().numpy()))

#     #     val_prs = np.stack(val_prs, 0)
#     #     val_gts = np.stack(val_gts, 0)

#     #     all_plcc = 0
#     #     for i, key in zip(range(16), max_gts_train.columns):
#     #         srcc, plcc = spearmanr(val_prs[:, i], val_gts[:, i])[0], pearsonr(val_prs[:, i], val_gts[:, i])[0]
#     #         print(key, srcc, plcc)
#     #         all_plcc += plcc
        
#     #     # Check for improvement and implement early stopping
#     #     if all_plcc > best_all_plcc:
#     #         with open("maxvqa_validation_results.pkl", "wb") as f:
#     #             pkl.dump(val_prs, f)
#     #         best_all_plcc = all_plcc
#     #         torch.save(maxvqa_ema.state_dict(), f"maxvqa_best_chunk_{chunk_num + 1}.pt")
#     #         no_improve_epochs = 0
#     #     else:
#     #         no_improve_epochs += 1
#     #         print(f"No improvement for {no_improve_epochs} epoch(s)")
            
#     #     # Early stopping condition
#     #     if no_improve_epochs >= early_stopping_patience:
#     #         print("Early stopping triggered")
#     #         break

'''#Implementing cosineAnnealingwarmrestarts
import os
import glob
import pickle as pkl
import random
import open_clip
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import time
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Define your loss functions
def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def count_parameters(model):
    for name, module in model.named_children():
        print(name, "|", sum(p.numel() for p in module.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MaxVisualFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, visual_features, max_gts, indices=None):
        super().__init__()
        if indices is None:
            indices = range(len(visual_features))
            print("Using all indices:", indices)
        self.visual_features = [visual_features[ind] for ind in indices]
        self.gts = [max_gts.iloc[ind].values for ind in indices]

    def __getitem__(self, index):
        return self.visual_features[index], torch.Tensor(self.gts[index])

    def __len__(self):
        return len(self.gts)

def encode_text_prompts(prompts, device="cuda"):
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        embedding = model.token_embedding(text_tokens)
        text_features = model.encode_text(text_tokens).float()
    return text_tokens, embedding, text_features

# You need to install DOVER
from dover import datasets
from dover import DOVER
import wandb
from model import TextEncoder, MaxVQA, EnhancedVisualEncoder

device = "cuda"

# Initialize datasets
with open("maxvqa.yml", "r") as f:
    opt = yaml.safe_load(f)

val_datasets = {}
for name, dataset in opt["data"].items():
    val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

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
text_encoder = TextEncoder(model).to(device)
visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder).to(device)
maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# Read the ground truth values from CSV file
max_gts_train = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_train.csv")
max_gts_val = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_val.csv")  # Assuming separate validation set CSV

# Ensure the model parameters are counted
print(f'The model has {count_parameters(maxvqa):,} trainable parameters')
optimizer = torch.optim.AdamW(maxvqa.parameters(), lr=1e-3)

# Initialize the scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)

# Make sure this directory exists to store extracted features
os.makedirs("features", exist_ok=True)

# Path for validation features
val_feat_path = "features/maxvqa_vis_val-maxwell.pkl"

# Extract or load validation features
if glob.glob(val_feat_path):
    print("Found pre-extracted visual features for validation set...")
    s = time.time()
    val_feats = torch.load(val_feat_path)
    print(f"Successfully loaded validation set features, elapsed {time.time() - s:.2f}s.")
else:
    print("Extracting features for validation set...")
    val_feats = []
    val_loader = torch.utils.data.DataLoader(
        val_datasets["val-maxwell"], batch_size=1, num_workers=8, pin_memory=True
    )
    for i, data in enumerate(tqdm(val_loader, desc="Extracting features for validation set")):
        if i >= 909:  # Limit to first 909 samples
            break
        with torch.no_grad():
            vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
            val_feats.append(vis_feats.half().cpu())
        torch.cuda.empty_cache()
    
    torch.save(val_feats, val_feat_path)
    print(f"Validation features saved to {val_feat_path}")

# Create the validation dataset and dataloader
test_dataset = MaxVisualFeatureDataset(val_feats, max_gts_val)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

maxvqa_ema = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# Initialize WandB
run = wandb.init(
    project="MaxVQA",
    name=f"maxvqa_maxwell_pushed",
    reinit=True,
    settings=wandb.Settings(start_method="thread"),
)

# Define chunks for training data
chunk_sizes = [912, 912, 912, 896]
num_chunks = len(chunk_sizes)
best_all_plcc = 0

train_losses = []
val_losses = []

for chunk_num in range(num_chunks):
    s = chunk_num * 912
    e = s + chunk_sizes[chunk_num]
    indices = range(s, e)
    print(f"Processing chunk {chunk_num + 1}/{num_chunks}: {s}-{e}")

    feat_path = f"features/maxvqa_vis_train_chunk_{chunk_num + 1}.pkl"
    if glob.glob(feat_path):
        print(f"Found pre-extracted visual features for chunk {chunk_num + 1}...")
        s = time.time()
        feats_chunk = torch.load(feat_path)
        print(f"Successfully loaded chunk {chunk_num + 1}, elapsed {time.time() - s:.2f}s.")
    else:
        print(f"Extracting features for training chunk {chunk_num + 1}...")
        feats_chunk = []
        for i in tqdm(indices, desc=f"Extracting features [{s}-{e}]"):
            data = val_datasets["train-maxwell"][i]  # Assuming "train-maxwell" is the dataset key
            with torch.no_grad():
                vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
                feats_chunk.append(vis_feats.half().cpu())
            torch.cuda.empty_cache()
        torch.save(feats_chunk, feat_path)
        print(f"Features for chunk {chunk_num + 1} saved to {feat_path}")

    # Create the train dataset and dataloader for this chunk
    train_dataset_chunk = MaxVisualFeatureDataset(feats_chunk, max_gts_train.iloc[indices])
    train_dataloader_chunk = torch.utils.data.DataLoader(train_dataset_chunk, batch_size=16, shuffle=True)

    # Training loop for this chunk
    maxvqa.train()
    for epoch in range(20):
        print(f"Epoch {epoch + 1}/20")
        epoch_loss = 0
        for data in tqdm(train_dataloader_chunk):
            optimizer.zero_grad()
            vis_feat, gt = data
            res = maxvqa(vis_feat.cuda(), text_encoder)
            loss, aux_loss = 0, 0
            for i in range(16):
                loss += plcc_loss(res[:, 0, i], gt[:, i].cuda().float())
                for j in range(i + 1, 16):
                    aux_loss += 0.005 * (0.5 - plcc_loss(res[:, 0, i], res[0, :, j]))
            wandb.log({"loss": loss.item(), "aux_loss": aux_loss.item()})
            loss += aux_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            model_params = dict(maxvqa.named_parameters())
            model_ema_params = dict(maxvqa_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999)

        # Append average training loss for the epoch
        train_losses.append(epoch_loss / len(train_dataloader_chunk))

        # Update learning rate with the scheduler
        scheduler.step()

        # Evaluation phase
        maxvqa.eval()
        val_loss = 0        
        val_prs, val_gts = [], []
        for data in tqdm(test_dataloader):
            with torch.no_grad():
                vis_feat, gt = data
                res = maxvqa_ema(vis_feat.cuda(), text_encoder)[:, 0]
                val_prs.extend(list(res.cpu().numpy()))
                val_gts.extend(list(gt.cpu().numpy()))
                # Calculate validation loss
                val_loss += plcc_loss(res, gt.cuda().float()).item()
        val_losses.append(val_loss / len(test_dataloader))

        val_prs = np.stack(val_prs, 0)
        val_gts = np.stack(val_gts, 0)

        all_plcc = 0
        for i, key in zip(range(16), max_gts_train):
            srcc, plcc = spearmanr(val_prs[:, i], val_gts[:, i])[0], pearsonr(val_prs[:, i], val_gts[:, i])[0]
            print(key, srcc, plcc)
            all_plcc += plcc

        if all_plcc > best_all_plcc:
            with open("maxvqa_validation_results.pkl", "wb") as f:
                pkl.dump(val_prs, f)
            best_all_plcc = all_plcc
            torch.save(maxvqa_ema.state_dict(), f"maxvqa_best_chunk_{chunk_num + 1}.pt")

# Plot learning curves after training
import matplotlib.pyplot as plt
print("Train Losses: ", len(train_losses))
print("Val Losses: ", len(val_losses))
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves (Loss)')
plt.savefig('results.png')
plt.show()
'''

#Adding Huber loss function
import os
import glob
import pickle as pkl
import random
import open_clip
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import time
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt

# Define your loss functions
def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def huber_loss(y_pred, y, delta=1.0):
    return nn.SmoothL1Loss()(y_pred, y)

def count_parameters(model):
    for name, module in model.named_children():
        print(name, "|", sum(p.numel() for p in module.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MaxVisualFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, visual_features, max_gts, indices=None):
        super().__init__()
        if indices is None:
            indices = range(len(visual_features))
            print("Using all indices:", indices)
        self.visual_features = [visual_features[ind] for ind in indices]
        self.gts = [max_gts.iloc[ind].values for ind in indices]

    def __getitem__(self, index):
        return self.visual_features[index], torch.Tensor(self.gts[index])

    def __len__(self):
        return len(self.gts)

def encode_text_prompts(prompts, device="cuda"):
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        embedding = model.token_embedding(text_tokens)
        text_features = model.encode_text(text_tokens).float()
    return text_tokens, embedding, text_features

# You need to install DOVER
from dover import datasets
from dover import DOVER
import wandb
from model import TextEncoder, MaxVQA, EnhancedVisualEncoder

device = "cuda"

# Initialize datasets
with open("maxvqa.yml", "r") as f:
    opt = yaml.safe_load(f)

val_datasets = {}
for name, dataset in opt["data"].items():
    val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

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
text_encoder = TextEncoder(model).to(device)
visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder).to(device)
maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# Read the ground truth values from CSV file
max_gts_train = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_train.csv")
max_gts_val = pd.read_csv("/home/ubuntu/MaxVQA/MaxWell_val.csv")  # Assuming separate validation set CSV

# Ensure the model parameters are counted
print(f'The model has {count_parameters(maxvqa):,} trainable parameters')
optimizer = torch.optim.AdamW(maxvqa.parameters(), lr=1e-3)

# Initialize the scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)

# Make sure this directory exists to store extracted features
os.makedirs("features", exist_ok=True)

# Path for validation features
val_feat_path = "features/maxvqa_vis_val-maxwell.pkl"

# Extract or load validation features
if glob.glob(val_feat_path):
    print("Found pre-extracted visual features for validation set...")
    s = time.time()
    val_feats = torch.load(val_feat_path)
    print(f"Successfully loaded validation set features, elapsed {time.time() - s:.2f}s.")
else:
    print("Extracting features for validation set...")
    val_feats = []
    val_loader = torch.utils.data.DataLoader(
        val_datasets["val-maxwell"], batch_size=1, num_workers=8, pin_memory=True
    )
    for i, data in enumerate(tqdm(val_loader, desc="Extracting features for validation set")):
        if i >= 909:  # Limit to first 909 samples
            break
        with torch.no_grad():
            vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
            val_feats.append(vis_feats.half().cpu())
        torch.cuda.empty_cache()
    
    torch.save(val_feats, val_feat_path)
    print(f"Validation features saved to {val_feat_path}")

# Create the validation dataset and dataloader
test_dataset = MaxVisualFeatureDataset(val_feats, max_gts_val)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

maxvqa_ema = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

# Initialize WandB
run = wandb.init(
    project="MaxVQA",
    name=f"maxvqa_maxwell_pushed",
    reinit=True,
    settings=wandb.Settings(start_method="thread"),
)

# Define chunks for training data
chunk_sizes = [912, 912, 912, 896]
num_chunks = len(chunk_sizes)
best_all_plcc = 0

train_losses = []
val_losses = []
huber_train_losses = []
huber_val_losses = []

for chunk_num in range(num_chunks):
    s = chunk_num * 912
    e = s + chunk_sizes[chunk_num]
    indices = range(s, e)
    print(f"Processing chunk {chunk_num + 1}/{num_chunks}: {s}-{e}")

    feat_path = f"features/maxvqa_vis_train_chunk_{chunk_num + 1}.pkl"
    if glob.glob(feat_path):
        print(f"Found pre-extracted visual features for chunk {chunk_num + 1}...")
        s = time.time()
        feats_chunk = torch.load(feat_path)
        print(f"Successfully loaded chunk {chunk_num + 1}, elapsed {time.time() - s:.2f}s.")
    else:
        print(f"Extracting features for training chunk {chunk_num + 1}...")
        feats_chunk = []
        for i in tqdm(indices, desc=f"Extracting features [{s}-{e}]"):
            data = val_datasets["train-maxwell"][i]  # Assuming "train-maxwell" is the dataset key
            with torch.no_grad():
                vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
                feats_chunk.append(vis_feats.half().cpu())
            torch.cuda.empty_cache()
        torch.save(feats_chunk, feat_path)
        print(f"Features for chunk {chunk_num + 1} saved to {feat_path}")

    # Create the train dataset and dataloader for this chunk
    train_dataset_chunk = MaxVisualFeatureDataset(feats_chunk, max_gts_train.iloc[indices])
    train_dataloader_chunk = torch.utils.data.DataLoader(train_dataset_chunk, batch_size=16, shuffle=True)

    # Training loop for this chunk
    maxvqa.train()
    for epoch in range(20):
        print(f"Epoch {epoch + 1}/20")
        epoch_loss = 0
        huber_epoch_loss = 0
        
        for data in tqdm(train_dataloader_chunk):
            optimizer.zero_grad()
            vis_feat, gt = data
            res = maxvqa(vis_feat.cuda(), text_encoder)
            
            # Compute individual losses
            plcc_loss_total, huber_loss_total, aux_loss_total = 0, 0, 0
            for i in range(16):
                plcc_loss_val = plcc_loss(res[:, 0, i], gt[:, i].cuda().float())
                huber_loss_val = huber_loss(res[:, 0, i], gt[:, i].cuda().float())
                plcc_loss_total += plcc_loss_val
                huber_loss_total += huber_loss_val
                for j in range(i + 1, 16):
                    aux_loss_val = 0.005 * (0.5 - plcc_loss(res[:, 0, i], res[0, :, j]))
                    aux_loss_total += aux_loss_val
            
            # Sum up all losses
            total_loss = plcc_loss_total + huber_loss_total + aux_loss_total
            
            # Log each loss separately
            wandb.log({"plcc_loss": plcc_loss_total.item(), "huber_loss": huber_loss_total.item(), "aux_loss": aux_loss_total.item(), "total_loss": total_loss.item()})
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            huber_epoch_loss += huber_loss_total.item()
            
            # Update EMA model parameters
            model_params = dict(maxvqa.named_parameters())
            model_ema_params = dict(maxvqa_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(model_params[k].data, alpha=1 - 0.999)

        # Append average training and huber loss for the epoch
        train_losses.append(epoch_loss / len(train_dataloader_chunk))
        huber_train_losses.append(huber_epoch_loss / len(train_dataloader_chunk))

        # Update learning rate with the scheduler
        scheduler.step()

        # Evaluation phase
        maxvqa.eval()
        val_loss = 0        
        huber_val_loss = 0
        val_prs, val_gts = [], []
        for data in tqdm(test_dataloader):
            with torch.no_grad():
                vis_feat, gt = data
                res = maxvqa_ema(vis_feat.cuda(), text_encoder)[:, 0]
                
                # Compute and log validation losses
                val_loss += plcc_loss(res, gt.cuda().float()).item()
                huber_val_loss += huber_loss(res, gt.cuda().float()).item()
                
                val_prs.extend(list(res.cpu().numpy()))
                val_gts.extend(list(gt.cpu().numpy()))
        
        val_losses.append(val_loss / len(test_dataloader))
        huber_val_losses.append(huber_val_loss / len(test_dataloader))
        
        # Log validation losses
        wandb.log({"validation_plcc_loss": val_loss, "validation_huber_loss": huber_val_loss})

        val_prs = np.stack(val_prs, 0)
        val_gts = np.stack(val_gts, 0)

        all_plcc = 0
        for i, key in zip(range(16), max_gts_train):
            srcc, plcc = spearmanr(val_prs[:, i], val_gts[:, i])[0], pearsonr(val_prs[:, i], val_gts[:, i])[0]
            print(key, srcc, plcc)
            all_plcc += plcc

        if all_plcc > best_all_plcc:
            with open("maxvqa_validation_results.pkl", "wb") as f:
                pkl.dump(val_prs, f)
            best_all_plcc = all_plcc
            torch.save(maxvqa_ema.state_dict(), f"maxvqa_best_chunk_{chunk_num + 1}.pt")

# Plot learning curves after training
print("Train Losses: ", len(train_losses))
print("Val Losses: ", len(val_losses))
print("Huber Train Losses: ", len(huber_train_losses))
print("Huber Val Losses: ", len(huber_val_losses))

# Plot training and validation losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves (Loss)')
plt.savefig('results.png')

# Plot Huber losses
plt.figure(figsize=(12, 4))
plt.plot(range(1, len(huber_train_losses) + 1), huber_train_losses, label='Huber Training Loss')
plt.plot(range(1, len(huber_val_losses) + 1), huber_val_losses, label='Huber Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Huber Loss')
plt.legend()
plt.title('Learning Curves (Huber Loss)')
plt.savefig('huber.png')

plt.show()