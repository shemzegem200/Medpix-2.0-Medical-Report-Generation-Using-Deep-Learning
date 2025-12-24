#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install torch torchvision transformers


# In[1]:


import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)

import torchvision.transforms as T
import torchvision.models as models

print("✅ torchvision fully loaded")


# In[2]:


import os
import json
import math
import random
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel


# In[3]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_REPORT_LEN = 120
EMBED_DIM = 256
TEXT_DIM = 768
KG_DIM = 256
HIDDEN_DIM = 512

BATCH_SIZE = 4
# NUM_WORKERS = 4
NUM_WORKERS = 0

print("Using device:", DEVICE)


# In[4]:


import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())


# In[5]:


image_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# In[6]:


LOCATION_TOKENS = {
    "Head": {
        "bos": "<HEAD_BOS>",
        "eos": "<HEAD_EOS>"
    },
    "Thorax": {
        "bos": "<THORAX_BOS>",
        "eos": "<THORAX_EOS>"
    },
    "Abdomen": {
        "bos": "<ABDOMEN_BOS>",
        "eos": "<ABDOMEN_EOS>"
    },
    "Spine and Muscles": {
        "bos": "<SPINE_BOS>",
        "eos": "<SPINE_EOS>"
    },
    "Reproductive and Urinary System": {
        "bos": "<GU_BOS>",
        "eos": "<GU_EOS>"
    }
}


# In[7]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

special_tokens = {
    "pad_token": "<PAD>",
    "additional_special_tokens": []
}

for loc in LOCATION_TOKENS:
    special_tokens["additional_special_tokens"].append(
        LOCATION_TOKENS[loc]["bos"]
    )
    special_tokens["additional_special_tokens"].append(
        LOCATION_TOKENS[loc]["eos"]
    )

tokenizer.add_special_tokens(special_tokens)

VOCAB_SIZE = len(tokenizer)
print("Vocab size:", VOCAB_SIZE)


# In[8]:


import ast
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MedPixDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Normalize NaNs early (VERY IMPORTANT)
        self.df = self.df.fillna("")

    def __len__(self):
        return len(self.df)

    def load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def parse_image_list(self, s):
        # CSV stores lists as strings: "['path1', 'path2']"
        if s == "" or s == "[]":
            return []
        return ast.literal_eval(s)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------- Images --------
        ct_images = []
        mri_images = []

        ct_paths = self.parse_image_list(row["CT_image_paths"])
        mri_paths = self.parse_image_list(row["MRI_image_paths"])

        for p in ct_paths:
            ct_images.append(self.load_image(p))

        for p in mri_paths:
            mri_images.append(self.load_image(p))

        # -------- Text Encoder Input --------
        text_input = row["combined_text"]

        # -------- Target Report --------
        report = row["findings"]

        # -------- Location (KG routing) --------
        location_category = row["Location Category"]

        return {
            "uid": row["U_id"],
            "ct_images": ct_images,     # list[Tensor]
            "mri_images": mri_images,   # list[Tensor]
            "text_input": text_input,   # str
            "report": report,           # str
            "location": location_category
        }


# In[9]:


def collate_fn(batch):
    """
    Batch items contain:
    - ct_images: list[Tensor]
    - mri_images: list[Tensor]
    - text_input: str
    - report: str
    - location: str
    """

    # ---------- CT images ----------
    max_ct = max(len(b["ct_images"]) for b in batch)
    ct_imgs, ct_masks = [], []

    for b in batch:
        imgs = b["ct_images"]
        if len(imgs) == 0:
            dummy = torch.zeros(3, 224, 224)
            imgs = [dummy]

        pad = max_ct - len(imgs)
        imgs = imgs + [torch.zeros_like(imgs[0])] * pad
        mask = [1] * (len(imgs) - pad) + [0] * pad

        ct_imgs.append(torch.stack(imgs))
        ct_masks.append(torch.tensor(mask))

    ct_imgs = torch.stack(ct_imgs)      # (B, N_ct, 3, H, W)
    ct_masks = torch.stack(ct_masks)    # (B, N_ct)

    # ---------- MRI images ----------
    max_mri = max(len(b["mri_images"]) for b in batch)
    mri_imgs, mri_masks = [], []

    for b in batch:
        imgs = b["mri_images"]
        if len(imgs) == 0:
            dummy = torch.zeros(3, 224, 224)
            imgs = [dummy]

        pad = max_mri - len(imgs)
        imgs = imgs + [torch.zeros_like(imgs[0])] * pad
        mask = [1] * (len(imgs) - pad) + [0] * pad

        mri_imgs.append(torch.stack(imgs))
        mri_masks.append(torch.tensor(mask))

    mri_imgs = torch.stack(mri_imgs)    # (B, N_mri, 3, H, W)
    mri_masks = torch.stack(mri_masks)  # (B, N_mri)

    # ---------- Text encoder input ----------
    text_inputs = [b["text_input"] for b in batch]
    text_enc = tokenizer(
        text_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # ---------- Decoder target ----------
    reports = [b["report"] for b in batch]   # <-- RAW ground truth
    report_enc = tokenizer(
        reports,
        padding=True,
        truncation=True,
        max_length=MAX_REPORT_LEN,
        return_tensors="pt"
    )

    # ---------- Location ----------
    locations = [b["location"] for b in batch]

    return {
        "ct_images": ct_imgs,
        "ct_masks": ct_masks,
        "mri_images": mri_imgs,
        "mri_masks": mri_masks,
        "text_input_ids": text_enc["input_ids"],
        "text_attention_mask": text_enc["attention_mask"],
        "report_input_ids": report_enc["input_ids"],
        "report_attention_mask": report_enc["attention_mask"],
        "reports": reports,              # <-- RAW ground-truth strings
        "locations": locations
    }


# In[10]:


ds = MedPixDataset(r"C:\fyp_manish_shyam_phase2\data\df_overall.csv", transform=image_transform)

sample = ds[0]
print(sample["uid"])
print("CT images:", len(sample["ct_images"]))
print("MRI images:", len(sample["mri_images"]))
print("Text length:", len(sample["text_input"]))
print("Report length:", len(sample["report"]))
print("Location:", sample["location"])


# In[11]:


import numpy as np
print(np.__version__)


# In[12]:


# Taking 2–3 samples manually for doing a small sanity check
batch_samples = [ds[i] for i in range(3)]

batch = collate_fn(batch_samples)
print("CT images shape:", batch["ct_images"].shape)
print("CT masks shape:", batch["ct_masks"].shape)

print("MRI images shape:", batch["mri_images"].shape)
print("MRI masks shape:", batch["mri_masks"].shape)

print("Text input ids shape:", batch["text_input_ids"].shape)
print("Report input ids shape:", batch["report_input_ids"].shape)

print("Raw reports count:", len(batch["reports"]))
print("First report preview:\n", batch["reports"][0][:200])

print("Locations:", batch["locations"])


# In[13]:


from torch.utils.data import DataLoader, random_split

# ---- Load full dataset ----
full_dataset = MedPixDataset(
    r"C:\fyp_manish_shyam_phase2\data\df_overall.csv",
    transform=image_transform
)

# ---- 80 / 20 split ----
dataset_size = len(full_dataset)
train_size = int(0.95 * dataset_size)
test_size = dataset_size - train_size

# Reproducibility
generator = torch.Generator().manual_seed(42)

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=generator
)

print(f"Total samples: {dataset_size}")
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples:  {len(test_dataset)}")


# In[14]:


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
    # , pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # No shuffle for test
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
    # , pin_memory=True
)

print("Train & Test loaders ready")


# In[15]:


import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for p in base.parameters():
            p.requires_grad = False

        base.fc = nn.Linear(base.fc.in_features, embed_dim)
        self.cnn = base

    def forward(self, x):
        """
        x: (B, N, 3, H, W)
        return: (B, N, D)
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, N, -1)
        return feats


# In[16]:


ct_encoder = ImageEncoder().to(DEVICE)
mri_encoder = ImageEncoder().to(DEVICE)


# In[17]:


def masked_mean_pooling(feats, masks):
    masks = masks.unsqueeze(-1).float()   # (B, N, 1)
    summed = (feats * masks).sum(dim=1)
    denom = masks.sum(dim=1).clamp(min=1e-6)
    return summed / denom


# In[18]:


# from transformers import AutoModel

# class TextEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained("bert-base-uncased")
#         self.proj = nn.Linear(TEXT_DIM, EMBED_DIM)

#     def forward(self, input_ids, attention_mask):
#         out = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         cls = out.last_hidden_state[:, 0]  # CLS token
#         return self.proj(cls)


from transformers import AutoModel
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, model_name, embed_dim):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)

        hidden_dim = self.lm.config.hidden_size
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, input_ids, attention_mask):

        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )

        # Mean pooling over tokens (causal models do NOT have CLS)
        hidden = outputs.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return self.proj(pooled)


# In[19]:


TEXT_MODEL_NAME = "bert-base-uncased"

text_encoder = TextEncoder(
    model_name=TEXT_MODEL_NAME,
    embed_dim=EMBED_DIM
).to(DEVICE)

text_encoder.lm.resize_token_embeddings(len(tokenizer))


# In[20]:


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, A_hat, X):
        """
        A_hat: (N, N) normalized adjacency
        X: (N, D)
        """
        return F.relu(self.linear(A_hat @ X))


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(GCNLayer(dims[i], dims[i + 1]))

        self.layers = nn.ModuleList(layers)

    def forward(self, A_hat, X):
        for layer in self.layers:
            X = layer(A_hat, X)
        return X.mean(dim=0)   # graph-level embedding


# In[21]:


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    A: (N, N) raw adjacency matrix
    returns: (N, N) normalized adjacency with self-loops
    """
    device = A.device
    N = A.size(0)

    # Add self-loops
    A_tilde = A + torch.eye(N, device=device)

    # Degree
    D = A_tilde.sum(dim=1)

    # D^{-1/2}
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(D_inv_sqrt)

    # Symmetric normalization
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_hat


# In[22]:


import pandas as pd
import numpy as np

KG_LOCATION_MAP = {
    "Head": r"C:\fyp_manish_shyam_phase2\data\split_by_location_category_matrices\Head_matrix.csv",
    "Thorax": r"C:\fyp_manish_shyam_phase2\data\split_by_location_category_matrices\Thorax_matrix.csv",
    "Abdomen": r"C:\fyp_manish_shyam_phase2\data\split_by_location_category_matrices\Abdomen_matrix.csv",
    "Spine and Muscles": r"C:\fyp_manish_shyam_phase2\data\split_by_location_category_matrices\Spine_and_Muscles_matrix.csv",
    "Reproductive and Urinary System": r"C:\fyp_manish_shyam_phase2\data\split_by_location_category_matrices\Reproductive_and_Urinary_System_matrix.csv"
}


A_hat_dict = {}

# ---- Load and normalize adjacency matrices ----
for loc, path in KG_LOCATION_MAP.items():
    df = pd.read_csv(path, index_col=0)

    A = torch.tensor(
        df.values,
        dtype=torch.float32,
        device=DEVICE
    )

    A_hat = normalize_adjacency(A)
    A_hat_dict[loc] = A_hat

    print(f"{loc}: A_hat shape = {A_hat.shape}")

# ---- Create shared X_nodes (identity) ----
# Node count inferred from any adjacency matrix
example_loc = next(iter(A_hat_dict))
N_nodes = A_hat_dict[example_loc].shape[0]

X_nodes = torch.eye(N_nodes, device=DEVICE)

print("Shared X_nodes shape:", X_nodes.shape)



# In[23]:


class FeatureFusion(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Linear(embed_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, ct_feat, mri_feat, text_feat, kg_feat):
        """
        All inputs: (B, EMBED_DIM)
        Output: (B, HIDDEN_DIM)
        """
        fused = torch.cat(
            [ct_feat, mri_feat, text_feat, kg_feat],
            dim=-1
        )
        fused = self.dropout(fused)
        return self.fc(fused)


# In[24]:


fusion = FeatureFusion().to(DEVICE)


# In[25]:


def get_kg_embeddings(locations, gcn, X_nodes, A_hat_dict):
    """
    locations: list[str], length B
    returns: (B, KG_DIM)
    """

    device = X_nodes.device

    # 1. Compute KG embedding ONCE per unique location
    unique_locations = set(locations)
    location_to_embedding = {}

    for loc in unique_locations:
        A_hat = A_hat_dict[loc]              # (N, N)
        kg_emb = gcn(A_hat, X_nodes)         # (KG_DIM,)
        location_to_embedding[loc] = kg_emb

    # 2. Assign embedding to each sample
    kg_embeds = [
        location_to_embedding[loc] for loc in locations
    ]

    return torch.stack(kg_embeds).to(device)   # (B, KG_DIM)


# In[26]:


class ReportDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, fused_feat, input_ids):
        """
        fused_feat: (B, HIDDEN_DIM)
        input_ids: (B, T)
        """

        emb = self.embedding(input_ids)          # (B, T, D)

        h0 = fused_feat.unsqueeze(0)             # (1, B, H)
        c0 = torch.zeros_like(h0)                # (1, B, H)

        out, _ = self.lstm(emb, (h0, c0))
        logits = self.fc(out)                    # (B, T, vocab)

        return logits


# In[27]:


decoder = ReportDecoderLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)


# In[28]:


KG_IN_DIM = X_nodes.shape[1]   # number of node features
KG_HIDDEN_DIM = 256            # you can tune this
KG_OUT_DIM = EMBED_DIM         # must match fusion input

gcn = GCN(
    in_dim=KG_IN_DIM,
    hidden_dim=KG_HIDDEN_DIM,
    out_dim=KG_OUT_DIM,
    num_layers=2
).to(DEVICE)

print("GCN initialized")


# In[29]:


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


# In[30]:


# Freeze all encoders and fusion
freeze_module(ct_encoder)
freeze_module(mri_encoder)
freeze_module(text_encoder)
freeze_module(gcn)
freeze_module(fusion)

# Ensure decoder is trainable
for p in decoder.parameters():
    p.requires_grad = True


# In[31]:


criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id
)

params = [p for p in decoder.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    params,
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)


# In[32]:


def count_trainable(name, module):
    n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{name}: {n:,} trainable params")

count_trainable("CT Encoder", ct_encoder)
count_trainable("MRI Encoder", mri_encoder)
count_trainable("Text Encoder", text_encoder)
count_trainable("GCN", gcn)
count_trainable("Fusion", fusion)
count_trainable("Decoder", decoder)


# In[33]:


# from tqdm import tqdm

# def train_one_epoch(train_loader):
#     ct_encoder.train()
#     mri_encoder.train()
#     text_encoder.train()
#     gcn.train()
#     fusion.train()
#     decoder.train()

#     total_loss = 0.0

#     pbar = tqdm(
#         train_loader,
#         desc="Training",
#         total=len(train_loader),
#         leave=True
#     )

#     for batch_idx, batch in enumerate(pbar):
#         optimizer.zero_grad()

#         # ---- Move tensors to device ----
#         ct_imgs = batch["ct_images"].to(DEVICE)
#         ct_masks = batch["ct_masks"].to(DEVICE)

#         mri_imgs = batch["mri_images"].to(DEVICE)
#         mri_masks = batch["mri_masks"].to(DEVICE)

#         text_ids = batch["text_input_ids"].to(DEVICE)
#         text_mask = batch["text_attention_mask"].to(DEVICE)

#         report_ids = batch["report_input_ids"].to(DEVICE)
#         locations = batch["locations"]

#         # ---- Image encoders ----
#         ct_feats = ct_encoder(ct_imgs)
#         mri_feats = mri_encoder(mri_imgs)

#         ct_pooled = masked_mean_pooling(ct_feats, ct_masks)
#         mri_pooled = masked_mean_pooling(mri_feats, mri_masks)

#         # ---- Text encoder ----
#         text_feat = text_encoder(text_ids, text_mask)

#         # ---- KG encoder (location-specific) ----
#         kg_feat = get_kg_embeddings(
#             locations, gcn, X_nodes, A_hat_dict
#         )

#         # ---- Fusion ----
#         fused_feat = fusion(
#             ct_pooled, mri_pooled, text_feat, kg_feat
#         )

#         # ---- Decoder (teacher forcing) ----
#         logits = decoder(
#             fused_feat,
#             report_ids[:, :-1]
#         )

#         targets = report_ids[:, 1:]

#         loss = criterion(
#             logits.reshape(-1, VOCAB_SIZE),
#             targets.reshape(-1)
#         )

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         # ---- Update progress bar ----
#         pbar.set_postfix(
#             loss=f"{loss.item():.4f}"
#         )

#     scheduler.step()

#     return total_loss / len(train_loader)



# from tqdm import tqdm

# def train_one_epoch(train_loader):
#     ct_encoder.train()
#     mri_encoder.train()
#     text_encoder.train()
#     gcn.train()
#     fusion.train()
#     decoder.train()
#     prefix_proj.train()

#     total_loss = 0.0

#     pbar = tqdm(
#         train_loader,
#         desc="Training",
#         total=len(train_loader),
#         leave=True
#     )

#     for batch_idx, batch in enumerate(pbar):
#         optimizer.zero_grad()

#         # ---- Move tensors to device ----
#         ct_imgs = batch["ct_images"].to(DEVICE)
#         ct_masks = batch["ct_masks"].to(DEVICE)

#         mri_imgs = batch["mri_images"].to(DEVICE)
#         mri_masks = batch["mri_masks"].to(DEVICE)

#         text_ids = batch["text_input_ids"].to(DEVICE)
#         text_mask = batch["text_attention_mask"].to(DEVICE)

#         report_ids = batch["report_input_ids"].to(DEVICE)
#         report_mask = batch["report_attention_mask"].to(DEVICE)

#         locations = batch["locations"]

#         # ---- Image encoders ----
#         ct_feats = ct_encoder(ct_imgs)
#         mri_feats = mri_encoder(mri_imgs)

#         ct_pooled = masked_mean_pooling(ct_feats, ct_masks)
#         mri_pooled = masked_mean_pooling(mri_feats, mri_masks)

#         # ---- Text encoder ----
#         text_feat = text_encoder(text_ids, text_mask)

#         # ---- KG encoder ----
#         kg_feat = get_kg_embeddings(
#             locations, gcn, X_nodes, A_hat_dict
#         )

#         # ---- Fusion ----
#         fused_feat = fusion(
#             ct_pooled, mri_pooled, text_feat, kg_feat
#         )  # (B, EMBED_DIM)

#         # ======================================================
#         # 🔑 BioGPT decoder with PREFIX CONDITIONING (CORRECT)
#         # ======================================================

#         # ---- Project fused features to prefix ----
#         prefix = prefix_proj(fused_feat).unsqueeze(1)  # (B, 1, H)

#         # ======================================================
#         # 🔑 Inject location-specific BOS tokens (MANUAL)
#         # ======================================================

#         B = report_ids.size(0)

#         # Convert location → BOS token id
#         bos_ids = [
#             tokenizer.convert_tokens_to_ids(
#                 LOCATION_TOKENS[loc]["bos"]
#             )
#             for loc in locations
#         ]

#         bos_ids = torch.tensor(
#             bos_ids,
#             device=report_ids.device
#         ).unsqueeze(1)   # (B, 1)

#         # Prepend BOS to report_ids
#         report_ids = torch.cat([bos_ids, report_ids], dim=1)

#         # Update report mask
#         bos_mask = torch.ones(
#             (B, 1),
#             device=report_mask.device
#         )
#         report_mask = torch.cat([bos_mask, report_mask], dim=1)


#         # ---- Token embeddings ----
#         token_embeds = decoder.get_input_embeddings()(report_ids)
#         inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

#         # ---- Attention mask (add prefix mask) ----
#         prefix_mask = torch.ones(
#             (report_mask.size(0), 1),
#             device=report_mask.device
#         )
#         attention_mask = torch.cat([prefix_mask, report_mask], dim=1)

#         # ======================================================
#         # 🔥 CORRECT LABEL SHIFT (THIS IS THE KEY FIX)
#         # ======================================================

#         # Clone report ids
#         labels = report_ids.clone()

#         # Ignore padding tokens
#         labels[labels == tokenizer.pad_token_id] = -100

#         # Add dummy label for prefix position
#         prefix_labels = torch.full(
#             (labels.size(0), 1),
#             -100,
#             device=labels.device
#         )

#         # Final labels align with inputs_embeds
#         labels = torch.cat([prefix_labels, labels], dim=1)

#         # ---- BioGPT forward ----
#         outputs = decoder(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             labels=labels
#         )

#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         pbar.set_postfix(loss=f"{loss.item():.4f}")

#     scheduler.step()

#     return total_loss / len(train_loader)



from tqdm import tqdm

def train_one_epoch(train_loader):
    decoder.train()

    ct_encoder.eval()
    mri_encoder.eval()
    text_encoder.eval()
    gcn.eval()
    fusion.eval()


    total_loss = 0.0

    pbar = tqdm(
        train_loader,
        desc="Training",
        total=len(train_loader),
        leave=True
    )

    for batch in pbar:
        optimizer.zero_grad()

        # =========================
        # Move tensors
        # =========================
        ct_imgs = batch["ct_images"].to(DEVICE)
        ct_masks = batch["ct_masks"].to(DEVICE)

        mri_imgs = batch["mri_images"].to(DEVICE)
        mri_masks = batch["mri_masks"].to(DEVICE)

        text_ids = batch["text_input_ids"].to(DEVICE)
        text_mask = batch["text_attention_mask"].to(DEVICE)

        report_ids = batch["report_input_ids"].to(DEVICE)
        locations = batch["locations"]

        # =========================
        # Encode modalities
        # =========================
        ct_feats = ct_encoder(ct_imgs)
        mri_feats = mri_encoder(mri_imgs)

        ct_pooled = masked_mean_pooling(ct_feats, ct_masks)
        mri_pooled = masked_mean_pooling(mri_feats, mri_masks)

        text_feat = text_encoder(text_ids, text_mask)

        kg_feat = get_kg_embeddings(
            locations, gcn, X_nodes, A_hat_dict
        )

        fused_feat = fusion(
            ct_pooled, mri_pooled, text_feat, kg_feat
        )  # (B, HIDDEN_DIM)

        # ======================================================
        # 🔑 Inject LOCATION-SPECIFIC BOS (PRESERVED)
        # ======================================================
        B = report_ids.size(0)

        bos_ids = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(
                    LOCATION_TOKENS[loc]["bos"]
                )
                for loc in locations
            ],
            device=report_ids.device
        ).unsqueeze(1)  # (B, 1)

        # Prepend BOS to reports
        report_ids = torch.cat([bos_ids, report_ids], dim=1)

        # ======================================================
        # 🔥 Teacher forcing (CORRECT SHIFT)
        # ======================================================
        decoder_inputs = report_ids[:, :-1]   # includes BOS
        targets = report_ids[:, 1:]           # next-token targets

        logits = decoder(
            fused_feat,
            decoder_inputs
        )  # (B, T, vocab)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    return total_loss / len(train_loader)



# In[34]:


# # 1. Same vocab size
# print(len(tokenizer), decoder.get_input_embeddings().weight.shape[0])

# # 2. Token → ID consistency
# tok = "<HEAD_BOS>"
# tid = tokenizer.convert_tokens_to_ids(tok)
# print(tok, tid)

# # 3. Embedding lookup works
# emb = decoder.get_input_embeddings().weight[tid]
# print(emb.shape)


# In[35]:


NUM_EPOCHS = 50

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(train_loader)
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
    )


# In[ ]:





# In[ ]:





# In[50]:


# @torch.no_grad()
# def generate_report(fused_feat, location, max_len=MAX_REPORT_LEN):
#     """
#     fused_feat: (1, EMBED_DIM)
#     location: str
#     returns: generated report (str)
#     """

#     decoder.eval()

#     bos_token = LOCATION_TOKENS[location]["bos"]
#     eos_token = LOCATION_TOKENS[location]["eos"]

#     bos_id = tokenizer.convert_tokens_to_ids(bos_token)
#     eos_id = tokenizer.convert_tokens_to_ids(eos_token)

#     generated_ids = [bos_id]

#     for _ in range(max_len):
#         inp = torch.tensor(
#             generated_ids,
#             dtype=torch.long,
#             device=DEVICE
#         ).unsqueeze(0)   # (1, T)

#         logits = decoder(fused_feat, inp)      # (1, T, vocab)
#         next_id = logits[0, -1].argmax(dim=-1).item()

#         generated_ids.append(next_id)

#         if next_id == eos_iC:
#             break

#     return tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True
#     )



# @torch.no_grad()
# def generate_report(fused_feat, location, max_len=MAX_REPORT_LEN):
#     decoder.eval()
#     prefix_proj.eval()

#     # ---- Prefix ----
#     prefix = prefix_proj(fused_feat).unsqueeze(1)  # (1, 1, H)

#     # ---- BOS token ----
#     bos_token = LOCATION_TOKENS[location]["bos"]
#     bos_id = tokenizer.convert_tokens_to_ids(bos_token)

#     generated_ids = [bos_id]

#     for _ in range(max_len):
#         input_ids = torch.tensor(
#             generated_ids,
#             device=DEVICE
#         ).unsqueeze(0)  # (1, T)

#         token_embeds = decoder.get_input_embeddings()(input_ids)

#         inputs_embeds = torch.cat(
#             [prefix, token_embeds],
#             dim=1
#         )

#         attention_mask = torch.ones(
#             inputs_embeds.size()[:-1],
#             device=DEVICE
#         )

#         outputs = decoder(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask
#         )

#         next_token_logits = outputs.logits[0, -1]
#         next_id = torch.argmax(next_token_logits).item()

#         generated_ids.append(next_id)

#         if next_id == tokenizer.convert_tokens_to_ids(
#             LOCATION_TOKENS[location]["eos"]
#         ):
#             break

#     return tokenizer.decode(generated_ids, skip_special_tokens=True)


# @torch.no_grad()
# def generate_report(
#     fused_feat,
#     location,
#     max_len=MAX_REPORT_LEN,
#     min_len=20
# ):
#     """
#     fused_feat: (1, EMBED_DIM)
#     location: str
#     """

#     decoder.eval()
#     prefix_proj.eval()

#     # ---- Prefix conditioning ----
#     prefix = prefix_proj(fused_feat).unsqueeze(1)  # (1, 1, H)

#     # ---- Location-specific tokens ----
#     bos_id = tokenizer.convert_tokens_to_ids(
#         LOCATION_TOKENS[location]["bos"]
#     )
#     eos_id = tokenizer.convert_tokens_to_ids(
#         LOCATION_TOKENS[location]["eos"]
#     )

#     generated_ids = [bos_id]
#     word_count = 0   # counts generated tokens EXCLUDING BOS

#     while word_count < max_len:

#         input_ids = torch.tensor(
#             generated_ids,
#             dtype=torch.long,
#             device=DEVICE
#         ).unsqueeze(0)  # (1, T)

#         # ---- Token embeddings ----
#         token_embeds = decoder.get_input_embeddings()(input_ids)

#         # ---- Prefix + tokens ----
#         inputs_embeds = torch.cat(
#             [prefix, token_embeds],
#             dim=1
#         )  # (1, 1+T, H)

#         # ---- Attention mask ----
#         attention_mask = torch.ones(
#             inputs_embeds.size()[:2],
#             device=DEVICE
#         )

#         outputs = decoder(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask
#         )

#         next_token_logits = outputs.logits[0, -1]
#         next_id = torch.argmax(next_token_logits).item()

#         # --------------------------------------------------
#         # 🚫 Ignore EOS if min_len not reached
#         # --------------------------------------------------
#         if next_id == eos_id and word_count < min_len:
#             continue

#         generated_ids.append(next_id)

#         # Count only real tokens (not BOS, not ignored EOS)
#         word_count += 1

#         # ---- Stop only if EOS AFTER min_len ----
#         if next_id == eos_iC:
#             break

#     return tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True
#     )



# import torch
# import torch.nn.functional as F

# @torch.no_grad()
# def generate_report(
#     fused_feat,
#     location,
#     max_len=MAX_REPORT_LEN,
#     min_len=20,
#     top_p=0.9,
#     temperature=1.0
# ):
#     """
#     fused_feat: (1, EMBED_DIM)
#     location: str
#     """

#     decoder.eval()
#     prefix_proj.eval()

#     # ---- Prefix conditioning ----
#     prefix = prefix_proj(fused_feat).unsqueeze(1)  # (1, 1, H)

#     # ---- Location-specific tokens ----
#     bos_id = tokenizer.convert_tokens_to_ids(
#         LOCATION_TOKENS[location]["bos"]
#     )
#     eos_id = tokenizer.convert_tokens_to_ids(
#         LOCATION_TOKENS[location]["eos"]
#     )

#     generated_ids = [bos_id]
#     word_count = 0

#     for _ in range(max_len * 2):  # safety cap

#         input_ids = torch.tensor(
#             generated_ids,
#             dtype=torch.long,
#             device=DEVICE
#         ).unsqueeze(0)

#         token_embeds = decoder.get_input_embeddings()(input_ids)

#         inputs_embeds = torch.cat(
#             [prefix, token_embeds],
#             dim=1
#         )

#         attention_mask = torch.ones(
#             inputs_embeds.size()[:2],
#             device=DEVICE
#         )

#         outputs = decoder(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask
#         )

#         logits = outputs.logits[0, -1] / temperature
#         probs = F.softmax(logits, dim=-1)

#         # ==================================================
#         # 🔑 TOP-P (NUCLEUS) SAMPLING
#         # ==================================================
#         sorted_probs, sorted_indices = torch.sort(
#             probs, descending=True
#         )
#         cumulative_probs = torch.cumsum(sorted_probs, dim=0)

#         # Keep smallest set whose cumulative prob >= top_p
#         cutoff = cumulative_probs > top_p
#         cutoff[1:] = cutoff[:-1].clone()
#         cutoff[0] = False

#         sorted_probs[cutoff] = 0.0
#         sorted_probs = sorted_probs / sorted_probs.sum()

#         next_id = torch.multinomial(sorted_probs, 1).item()
#         next_id = sorted_indices[next_id].item()

#         # ---- Ignore early EOS ----
#         if next_id == eos_id and word_count < min_len:
#             continue

#         generated_ids.append(next_id)
#         word_count += 1

#         if next_id == eos_iC:
#             break

#         if word_count >= max_len:
#             break

#     return tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True
#     )


# import torch
# import torch.nn.functional as F

# @torch.no_grad()
# def generate_report(
#     fused_feat,
#     location,
#     max_len=MAX_REPORT_LEN,
#     min_len=20
# ):
#     """
#     fused_feat: (1, HIDDEN_DIM)
#     location: str
#     """

#     decoder.eval()

#     # ---- Location-specific tokens ----
#     bos_id = tokenizer.convert_tokens_to_ids(
#         LOCATION_TOKENS[location]["bos"]
#     )
#     eos_id = tokenizer.convert_tokens_to_ids(
#         LOCATION_TOKENS[location]["eos"]
#     )

#     generated_ids = [bos_id]
#     word_count = 0   # counts tokens EXCLUDING BOS

#     # safety cap (same as your original)
#     for _ in range(max_len * 2):

#         input_ids = torch.tensor(
#             generated_ids,
#             dtype=torch.long,
#             device=DEVICE
#         ).unsqueeze(0)   # (1, T)

#         # ---- GRU decoder forward ----
#         logits = decoder(
#             fused_feat,
#             input_ids
#         )  # (1, T, vocab)

#         logits = logits[0, -1] / temperature
#         probs = F.softmax(logits, dim=-1)

#         # # ==================================================
#         # # 🔑 TOP-P (NUCLEUS) SAMPLING (UNCHANGED)
#         # # ==================================================
#         # sorted_probs, sorted_indices = torch.sort(
#         #     probs, descending=True
#         # )
#         # cumulative_probs = torch.cumsum(sorted_probs, dim=0)

#         # cutoff = cumulative_probs > top_p
#         # cutoff[1:] = cutoff[:-1].clone()
#         # cutoff[0] = False

#         # sorted_probs[cutoff] = 0.0
#         # sorted_probs = sorted_probs / sorted_probs.sum()

#         next_id = torch.argmax(probs).item()

#         # next_id = sorted_indices[next_id].item()

#         # ---- Ignore early EOS ----
#         if next_id == eos_id and word_count < min_len:
#             continue

#         generated_ids.append(next_id)
#         word_count += 1

#         # ---- Stop conditions ----
#         if next_id == eos_iC:
#             break

#         if word_count >= max_len:
#             break

#     return tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True
#     )


@torch.no_grad()
def generate_report(
    fused_feat,
    location,
    max_len=MAX_REPORT_LEN,
    min_len=20
):
    decoder.eval()

    bos_id = tokenizer.convert_tokens_to_ids(
        LOCATION_TOKENS[location]["bos"]
    )
    eos_id = tokenizer.convert_tokens_to_ids(
        LOCATION_TOKENS[location]["eos"]
    )

    generated_ids = [bos_id]
    word_count = 0

    for _ in range(max_len * 2):

        input_ids = torch.tensor(
            generated_ids,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)   # (1, T)

        logits = decoder(
            fused_feat,
            input_ids
        )  # (1, T, vocab)

        next_id = torch.argmax(logits[0, -1]).item()

        if next_id == eos_id and word_count < min_len:
            continue

        generated_ids.append(next_id)
        word_count += 1

        if next_id == eos_id or word_count >= max_len:
            break

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )


# In[51]:


# from tqdm import tqdm

# @torch.no_grad()
# def run_inference(test_loader):
#     ct_encoder.eval()
#     mri_encoder.eval()
#     text_encoder.eval()
#     gcn.eval()
#     fusion.eval()
#     decoder.eval()

#     generated_reports = []
#     ground_truth_reports = []
#     locations_all = []

#     pbar = tqdm(
#         test_loader,
#         desc="Generating reports",
#         total=len(test_loader),
#         leave=True
#     )

#     for batch in pbar:
#         # ---- Move tensors ----
#         ct_imgs = batch["ct_images"].to(DEVICE)
#         ct_masks = batch["ct_masks"].to(DEVICE)

#         mri_imgs = batch["mri_images"].to(DEVICE)
#         mri_masks = batch["mri_masks"].to(DEVICE)

#         text_ids = batch["text_input_ids"].to(DEVICE)
#         text_mask = batch["text_attention_mask"].to(DEVICE)

#         report_ids = batch["report_input_ids"]   # keep on CPU for decoding GT
#         locations = batch["locations"]

#         # ---- Encode images ----
#         ct_feats = ct_encoder(ct_imgs)
#         mri_feats = mri_encoder(mri_imgs)

#         ct_pooled = masked_mean_pooling(ct_feats, ct_masks)
#         mri_pooled = masked_mean_pooling(mri_feats, mri_masks)

#         # ---- Encode text ----
#         text_feat = text_encoder(text_ids, text_mask)

#         # ---- KG embeddings ----
#         kg_feat = get_kg_embeddings(
#             locations, gcn, X_nodes, A_hat_dict
#         )

#         # ---- Fusion ----
#         fused_feats = fusion(
#             ct_pooled, mri_pooled, text_feat, kg_feat
#         )   # (B, EMBED_DIM)

#         # ---- Generate per sample ----
#         B = fused_feats.size(0)

#         for i in range(B):
#             gen_report = generate_report(
#                 fused_feats[i].unsqueeze(0),
#                 locations[i]
#             )

#             gt_report = tokenizer.decode(
#                 report_ids[i],
#                 skip_special_tokens=True
#             )

#             generated_reports.append(gen_report)
#             ground_truth_reports.append(gt_report)
#             locations_all.append(locations[i])

#     return generated_reports, ground_truth_reports, locations_all



# from tqdm import tqdm

# @torch.no_grad()
# def run_inference(test_loader):
#     ct_encoder.eval()
#     mri_encoder.eval()
#     text_encoder.eval()
#     gcn.eval()
#     fusion.eval()
#     decoder.eval()
#     prefix_proj.eval()   # 🔑 FIX 1

#     generated_reports = []
#     ground_truth_reports = []
#     locations_all = []

#     pbar = tqdm(
#         test_loader,
#         desc="Generating reports",
#         total=len(test_loader),
#         leave=True
#     )

#     for batch in pbar:
#         # ---- Move tensors ----
#         ct_imgs = batch["ct_images"].to(DEVICE)
#         ct_masks = batch["ct_masks"].to(DEVICE)

#         mri_imgs = batch["mri_images"].to(DEVICE)
#         mri_masks = batch["mri_masks"].to(DEVICE)

#         text_ids = batch["text_input_ids"].to(DEVICE)
#         text_mask = batch["text_attention_mask"].to(DEVICE)

#         report_ids = batch["report_input_ids"]  # CPU OK
#         locations = batch["locations"]

#         # ---- Encode images ----
#         ct_feats = ct_encoder(ct_imgs)
#         mri_feats = mri_encoder(mri_imgs)

#         ct_pooled = masked_mean_pooling(ct_feats, ct_masks)
#         mri_pooled = masked_mean_pooling(mri_feats, mri_masks)

#         # ---- Encode text ----
#         text_feat = text_encoder(text_ids, text_mask)

#         # ---- KG embeddings ----
#         kg_feat = get_kg_embeddings(
#             locations, gcn, X_nodes, A_hat_dict
#         )

#         # ---- Fusion ----
#         fused_feats = fusion(
#             ct_pooled, mri_pooled, text_feat, kg_feat
#         )   # (B, EMBED_DIM)

#         # ---- Generate per sample ----
#         for i in range(fused_feats.size(0)):

#             gen_report = generate_report(
#                 fused_feats[i].unsqueeze(0),
#                 locations[i]
#             )

#             # 🔑 FIX 2: align GT with training format
#             bos_token = LOCATION_TOKENS[locations[i]]["bos"]
#             gt_report = bos_token + " " + tokenizer.decode(
#                 report_ids[i],
#                 skip_special_tokens=True
#             )

#             generated_reports.append(gen_report)
#             ground_truth_reports.append(gt_report)
#             locations_all.append(locations[i])

#     return generated_reports, ground_truth_reports, locations_all


from tqdm import tqdm

@torch.no_grad()
def run_inference(test_loader):
    ct_encoder.eval()
    mri_encoder.eval()
    text_encoder.eval()
    gcn.eval()
    fusion.eval()
    decoder.eval()   # ✅ GRU decoder only

    generated_reports = []
    ground_truth_reports = []
    locations_all = []

    pbar = tqdm(
        test_loader,
        desc="Generating reports",
        total=len(test_loader),
        leave=True
    )

    for batch in pbar:
        # =========================
        # Move tensors
        # =========================
        ct_imgs = batch["ct_images"].to(DEVICE)
        ct_masks = batch["ct_masks"].to(DEVICE)

        mri_imgs = batch["mri_images"].to(DEVICE)
        mri_masks = batch["mri_masks"].to(DEVICE)

        text_ids = batch["text_input_ids"].to(DEVICE)
        text_mask = batch["text_attention_mask"].to(DEVICE)

        report_ids = batch["report_input_ids"]   # CPU OK
        locations = batch["locations"]

        # =========================
        # Encode modalities
        # =========================
        ct_feats = ct_encoder(ct_imgs)
        mri_feats = mri_encoder(mri_imgs)

        ct_pooled = masked_mean_pooling(ct_feats, ct_masks)
        mri_pooled = masked_mean_pooling(mri_feats, mri_masks)

        text_feat = text_encoder(text_ids, text_mask)

        kg_feat = get_kg_embeddings(
            locations, gcn, X_nodes, A_hat_dict
        )

        fused_feats = fusion(
            ct_pooled, mri_pooled, text_feat, kg_feat
        )   # (B, HIDDEN_DIM)

        # =========================
        # Generate per sample
        # =========================
        for i in range(fused_feats.size(0)):

            gen_report = generate_report(
                fused_feats[i].unsqueeze(0),
                locations[i]
            )

            # ==================================================
            # 🔑 Align GT format with training (KEEP THIS)
            # ==================================================
            bos_token = LOCATION_TOKENS[locations[i]]["bos"]
            gt_report = bos_token + " " + tokenizer.decode(
                report_ids[i],
                skip_special_tokens=True
            )

            generated_reports.append(gen_report)
            ground_truth_reports.append(gt_report)
            locations_all.append(locations[i])

    return generated_reports, ground_truth_reports, locations_all


# In[52]:


import pandas as pd

generated_reports, ground_truth_reports, locations_all = run_inference(test_loader)


# In[53]:


results_df = pd.DataFrame({
    "location": locations_all,
    "generated_report": generated_reports,
    "ground_truth_report": ground_truth_reports
})

save_path = r"C:\fyp_manish_shyam_phase2\results\generated_vs_gt_reports_decoder_only_unfrozen.csv"
results_df.to_csv(save_path, index=False)

print(f"Saved results to: {save_path}")
print("Total samples:", len(results_df))


# In[54]:


# =========================
# STAGE 2: Unfreeze Fusion
# =========================

# Keep encoders frozen
freeze_module(ct_encoder)
freeze_module(mri_encoder)
freeze_module(text_encoder)
freeze_module(gcn)

# Unfreeze fusion
for p in fusion.parameters():
    p.requires_grad = True

# Decoder already trainable
for p in decoder.parameters():
    p.requires_grad = True


# In[55]:


print("=== Trainable Parameters Check ===")
count_trainable("CT Encoder", ct_encoder)
count_trainable("MRI Encoder", mri_encoder)
count_trainable("Text Encoder", text_encoder)
count_trainable("GCN", gcn)
count_trainable("Fusion", fusion)
count_trainable("Decoder", decoder)


# In[56]:


# =========================
# Optimizer for Stage 2
# =========================

stage2_params = []

stage2_params += [p for p in fusion.parameters() if p.requires_grad]
stage2_params += [p for p in decoder.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    stage2_params,
    lr=1e-4,          # 🔑 LOWER LR
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

print("Stage-2 optimizer ready")


# In[57]:


from tqdm import tqdm

def train_one_epoch_stage2(train_loader):
    fusion.train()
    decoder.train()

    ct_encoder.eval()
    mri_encoder.eval()
    text_encoder.eval()
    gcn.eval()

    total_loss = 0.0

    pbar = tqdm(
        train_loader,
        desc="Stage-2 Training (Fusion + Decoder)",
        total=len(train_loader),
        leave=True
    )

    for batch in pbar:
        optimizer.zero_grad()

        # =========================
        # Move tensors
        # =========================
        ct_imgs = batch["ct_images"].to(DEVICE)
        ct_masks = batch["ct_masks"].to(DEVICE)

        mri_imgs = batch["mri_images"].to(DEVICE)
        mri_masks = batch["mri_masks"].to(DEVICE)

        text_ids = batch["text_input_ids"].to(DEVICE)
        text_mask = batch["text_attention_mask"].to(DEVICE)

        report_ids = batch["report_input_ids"].to(DEVICE)
        locations = batch["locations"]

        # =========================
        # Encode modalities
        # =========================
        with torch.no_grad():
            ct_feats = ct_encoder(ct_imgs)
            mri_feats = mri_encoder(mri_imgs)

            ct_pooled = masked_mean_pooling(ct_feats, ct_masks)
            mri_pooled = masked_mean_pooling(mri_feats, mri_masks)

            text_feat = text_encoder(text_ids, text_mask)

            kg_feat = get_kg_embeddings(
                locations, gcn, X_nodes, A_hat_dict
            )

        fused_feat = fusion(
            ct_pooled, mri_pooled, text_feat, kg_feat
        )

        # =========================
        # Location-specific BOS
        # =========================
        B = report_ids.size(0)

        bos_ids = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(
                    LOCATION_TOKENS[loc]["bos"]
                )
                for loc in locations
            ],
            device=report_ids.device
        ).unsqueeze(1)

        report_ids = torch.cat([bos_ids, report_ids], dim=1)

        decoder_inputs = report_ids[:, :-1]
        targets = report_ids[:, 1:]

        logits = decoder(fused_feat, decoder_inputs)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    return total_loss / len(train_loader)


# In[58]:


STAGE2_EPOCHS = 10

for epoch in range(STAGE2_EPOCHS):
    loss = train_one_epoch_stage2(train_loader)
    print(
        f"[Stage-2] Epoch {epoch+1}/{STAGE2_EPOCHS} | "
        f"Loss: {loss:.4f}"
    )


# In[59]:


import pandas as pd

generated_reports, ground_truth_reports, locations_all = run_inference(test_loader)


# In[60]:


results_df = pd.DataFrame({
    "location": locations_all,
    "generated_report": generated_reports,
    "ground_truth_report": ground_truth_reports
})

save_path = r"C:\fyp_manish_shyam_phase2\results\generated_vs_gt_reports_fusion_unfrozen.csv"
results_df.to_csv(save_path, index=False)

print(f"Saved results to: {save_path}")
print("Total samples:", len(results_df))


# In[61]:


# =========================
# STAGE 3: Unfreeze GCN
# =========================

# Keep encoders frozen
freeze_module(ct_encoder)
freeze_module(mri_encoder)
freeze_module(text_encoder)

# Unfreeze GCN
for p in gcn.parameters():
    p.requires_grad = True

# Fusion + Decoder remain trainable
for p in fusion.parameters():
    p.requires_grad = True

for p in decoder.parameters():
    p.requires_grad = True


# In[62]:


print("=== Stage-3 Trainable Params Check ===")
count_trainable("CT Encoder", ct_encoder)
count_trainable("MRI Encoder", mri_encoder)
count_trainable("Text Encoder", text_encoder)
count_trainable("GCN", gcn)
count_trainable("Fusion", fusion)
count_trainable("Decoder", decoder)


# In[63]:


# =========================
# Optimizer for Stage 3
# =========================

stage3_params = []

stage3_params += [p for p in gcn.parameters() if p.requires_grad]
stage3_params += [p for p in fusion.parameters() if p.requires_grad]
stage3_params += [p for p in decoder.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    stage3_params,
    lr=5e-5,          # 🔑 LOWER LR (KG is sensitive)
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

print("Stage-3 optimizer ready")


# In[64]:


from tqdm import tqdm

def train_one_epoch_stage3(train_loader):
    gcn.train()
    fusion.train()
    decoder.train()

    ct_encoder.eval()
    mri_encoder.eval()
    text_encoder.eval()

    total_loss = 0.0

    pbar = tqdm(
        train_loader,
        desc="Stage-3 Training (GCN + Fusion + Decoder)",
        total=len(train_loader),
        leave=True
    )

    for batch in pbar:
        optimizer.zero_grad()

        # =========================
        # Move tensors
        # =========================
        ct_imgs = batch["ct_images"].to(DEVICE)
        ct_masks = batch["ct_masks"].to(DEVICE)

        mri_imgs = batch["mri_images"].to(DEVICE)
        mri_masks = batch["mri_masks"].to(DEVICE)

        text_ids = batch["text_input_ids"].to(DEVICE)
        text_mask = batch["text_attention_mask"].to(DEVICE)

        report_ids = batch["report_input_ids"].to(DEVICE)
        locations = batch["locations"]

        # =========================
        # Encode frozen modalities
        # =========================
        with torch.no_grad():
            ct_feats = ct_encoder(ct_imgs)
            mri_feats = mri_encoder(mri_imgs)

            ct_pooled = masked_mean_pooling(ct_feats, ct_masks)
            mri_pooled = masked_mean_pooling(mri_feats, mri_masks)

            text_feat = text_encoder(text_ids, text_mask)

        # =========================
        # GCN now TRAINABLE
        # =========================
        kg_feat = get_kg_embeddings(
            locations, gcn, X_nodes, A_hat_dict
        )

        fused_feat = fusion(
            ct_pooled, mri_pooled, text_feat, kg_feat
        )

        # =========================
        # Location BOS + decoding
        # =========================
        bos_ids = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(
                    LOCATION_TOKENS[loc]["bos"]
                )
                for loc in locations
            ],
            device=report_ids.device
        ).unsqueeze(1)

        report_ids = torch.cat([bos_ids, report_ids], dim=1)

        decoder_inputs = report_ids[:, :-1]
        targets = report_ids[:, 1:]

        logits = decoder(fused_feat, decoder_inputs)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    return total_loss / len(train_loader)


# In[65]:


STAGE3_EPOCHS = 10

for epoch in range(STAGE3_EPOCHS):
    loss = train_one_epoch_stage3(train_loader)
    print(
        f"[Stage-3] Epoch {epoch+1}/{STAGE3_EPOCHS} | "
        f"Loss: {loss:.4f}"
    )


# In[66]:


import pandas as pd

generated_reports, ground_truth_reports, locations_all = run_inference(test_loader)


# In[67]:


results_df = pd.DataFrame({
    "location": locations_all,
    "generated_report": generated_reports,
    "ground_truth_report": ground_truth_reports
})

save_path = r"C:\fyp_manish_shyam_phase2\results\generated_vs_gt_reports_GCN_unfrozen.csv"
results_df.to_csv(save_path, index=False)

print(f"Saved results to: {save_path}")
print("Total samples:", len(results_df))


# In[68]:


# =========================
# Helper: Unfreeze ResNet layer4 only
# =========================

def unfreeze_resnet_layer4(resnet_model):
    # Freeze everything
    for p in resnet_model.parameters():
        p.requires_grad = False

    # Unfreeze layer4
    for p in resnet_model.layer4.parameters():
        p.requires_grad = True

    # Keep FC trainable (already replaced)
    for p in resnet_model.fc.parameters():
        p.requires_grad = True


# In[69]:


# =========================
# STAGE 4: Partial Image Unfreeze
# =========================

unfreeze_resnet_layer4(ct_encoder.cnn)
unfreeze_resnet_layer4(mri_encoder.cnn)

# Text encoder stays frozen
freeze_module(text_encoder)

# GCN + Fusion + Decoder remain trainable
for p in gcn.parameters():
    p.requires_grad = True

for p in fusion.parameters():
    p.requires_grad = True

for p in decoder.parameters():
    p.requires_grad = True


# In[70]:


print("=== Stage-4 Trainable Params Check ===")
count_trainable("CT Encoder", ct_encoder)
count_trainable("MRI Encoder", mri_encoder)
count_trainable("Text Encoder", text_encoder)
count_trainable("GCN", gcn)
count_trainable("Fusion", fusion)
count_trainable("Decoder", decoder)


# In[73]:


# =========================
# Optimizer for Stage 4
# =========================

stage4_params = []

stage4_params += [p for p in ct_encoder.parameters() if p.requires_grad]
stage4_params += [p for p in mri_encoder.parameters() if p.requires_grad]
stage4_params += [p for p in gcn.parameters() if p.requires_grad]
stage4_params += [p for p in fusion.parameters() if p.requires_grad]
stage4_params += [p for p in decoder.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    stage4_params,
    lr=1e-5,          # 🔑 VERY IMPORTANT: small LR
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

print("Stage-4 optimizer ready")


# In[75]:


from tqdm import tqdm

def train_one_epoch_stage4(train_loader):
    ct_encoder.train()
    mri_encoder.train()
    gcn.train()
    fusion.train()
    decoder.train()

    text_encoder.eval()   # still frozen

    total_loss = 0.0

    pbar = tqdm(
        train_loader,
        desc="Stage-4 Training (Visual + KG + Decoder)",
        total=len(train_loader),
        leave=True
    )

    for batch in pbar:
        optimizer.zero_grad()

        # =========================
        # Move tensors
        # =========================
        ct_imgs = batch["ct_images"].to(DEVICE)
        ct_masks = batch["ct_masks"].to(DEVICE)

        mri_imgs = batch["mri_images"].to(DEVICE)
        mri_masks = batch["mri_masks"].to(DEVICE)

        text_ids = batch["text_input_ids"].to(DEVICE)
        text_mask = batch["text_attention_mask"].to(DEVICE)

        report_ids = batch["report_input_ids"].to(DEVICE)
        locations = batch["locations"]

        # =========================
        # Encode modalities
        # =========================
        ct_feats = ct_encoder(ct_imgs)
        mri_feats = mri_encoder(mri_imgs)

        ct_pooled = masked_mean_pooling(ct_feats, ct_masks)
        mri_pooled = masked_mean_pooling(mri_feats, mri_masks)

        with torch.no_grad():
            text_feat = text_encoder(text_ids, text_mask)

        kg_feat = get_kg_embeddings(
            locations, gcn, X_nodes, A_hat_dict
        )

        fused_feat = fusion(
            ct_pooled, mri_pooled, text_feat, kg_feat
        )

        # =========================
        # Location BOS + decoding
        # =========================
        bos_ids = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(
                    LOCATION_TOKENS[loc]["bos"]
                )
                for loc in locations
            ],
            device=report_ids.device
        ).unsqueeze(1)

        report_ids = torch.cat([bos_ids, report_ids], dim=1)

        decoder_inputs = report_ids[:, :-1]
        targets = report_ids[:, 1:]

        logits = decoder(fused_feat, decoder_inputs)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    return total_loss / len(train_loader)


# In[76]:


STAGE4_EPOCHS = 6

for epoch in range(STAGE4_EPOCHS):
    loss = train_one_epoch_stage4(train_loader)
    print(
        f"[Stage-4] Epoch {epoch+1}/{STAGE4_EPOCHS} | "
        f"Loss: {loss:.4f}"
    )


# In[77]:


import pandas as pd

generated_reports, ground_truth_reports, locations_all = run_inference(test_loader)


# In[78]:


results_df = pd.DataFrame({
    "location": locations_all,
    "generated_report": generated_reports,
    "ground_truth_report": ground_truth_reports
})

save_path = r"C:\fyp_manish_shyam_phase2\results\generated_vs_gt_reports_encoder_unfrozen.csv"
results_df.to_csv(save_path, index=False)

print(f"Saved results to: {save_path}")
print("Total samples:", len(results_df))


# In[ ]:




