# filename: grpo_rlaif_train.py
# AMP FIXED for bfloat16 (PyTorch does not support scaler with bf16)

import os, re, math, random, time, json, codecs, warnings
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# CONFIG and paths
IN_CSV        = "./earnings_to_search.csv"
TRANSCRIPT_FN = lambda link: f".{link}.txt"
OUT_DIR       = "./grpo_student_ckpt"
os.makedirs(OUT_DIR, exist_ok=True)

TEACHER_MODEL_NAME = "yiyanghkust/finbert-tone"
STUDENT_BACKBONE   = "distilbert-base-uncased"

MAX_LENGTH = 32
PREP_REMARKS = True
SENT_CAP_PER_TRANSCRIPT = 400
PAIR_MARGIN = 0.10

DEV_LIMIT_ROWS = 100
VAL_SPLIT = 0.1

EPOCHS = 2
BATCH_SIZE = 64
LR = 3e-5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PAIR_TEMP = 1.0
SEED = 42

# Set seed and device
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if has_gpu else "cpu")
gpu_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
dtype = torch.float32  # use float32 to avoid BF16 mismatch issues
print("\U0001F527 Device:", gpu_name, "| dtype:", dtype)

# Load teacher model
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, use_fast=True)
teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME).to(device)
teacher_model.eval()

# Load student model
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BACKBONE, use_fast=True)
student_backbone = AutoModel.from_pretrained(STUDENT_BACKBONE).to(device)

class StudentModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)

student = StudentModel(student_backbone).to(device)
print("\U0001F4E6 Student initialized with scalar head.")

# Teacher scoring
@torch.inference_mode()
def teacher_p_pos(texts):
    enc = teacher_tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
    probs = torch.softmax(teacher_model(**enc).logits, dim=-1).float().cpu().numpy()
    pos_idx = {v.lower(): k for k, v in teacher_model.config.id2label.items()}.get("positive", 1)
    return probs[:, pos_idx]

# Data helpers
WORD_RE = re.compile(r"\b[a-zA-Z']+\b")
TICKER_RE = re.compile(r"\(([^)]+)\)")

def read_file(path):
    data, ticker = [], ""
    if not os.path.exists(path): return [], ""
    with codecs.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if not ticker and "start" not in line.lower():
                m = TICKER_RE.search(line)
                if m: ticker = m.group(1).split(":")[-1]
            data.append([s.strip() for s in re.split(r"\.\s+", line) if s.strip()])
    return data, ticker

def prepared_remarks(flat):
    start, out = False, []
    for s in flat:
        s_low = s.lower()
        if not start and (s_low.startswith("unknown") or "good" in s_low or "hello" in s_low): start = True
        if s_low.strip() == "question-and-answer session": break
        if start: out.append(s)
    return out or flat

def make_pairs(sents, ppos):
    n = len(sents)
    pairs = []
    for _ in range(min(4 * n, 5000)):
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i == j or abs(ppos[i] - ppos[j]) < PAIR_MARGIN: continue
        pairs.append((sents[i], sents[j], 1) if ppos[i] > ppos[j] else (sents[j], sents[i], 1))
    return pairs

# Dataset
class PairDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def collate(batch):
    a, b, lab = zip(*batch)
    a_enc = student_tokenizer(list(a), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
    b_enc = student_tokenizer(list(b), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
    return a_enc, b_enc, torch.tensor(lab, dtype=torch.float32, device=device)

def grpo_loss(logit_a, logit_b, temp=PAIR_TEMP):
    return torch.nn.functional.softplus(-(logit_a - logit_b)/temp).mean()

# Mining data
df = pd.read_csv(IN_CSV)
colmap = {c.lower(): c for c in df.columns}
rows = list(df.itertuples(index=False))[:DEV_LIMIT_ROWS]
pairs = []
print("\U0001F5C2️ Mining RLAIF preference pairs...")
for r in tqdm(rows):
    txt_path = TRANSCRIPT_FN(getattr(r, colmap['article']))
    data, _ = read_file(txt_path)
    if not data: continue
    flat = [s for block in data for s in block]
    sents = prepared_remarks(flat) if PREP_REMARKS else flat
    sents = sents[:SENT_CAP_PER_TRANSCRIPT] if SENT_CAP_PER_TRANSCRIPT else sents
    ppos = np.concatenate([teacher_p_pos(sents[i:i+128]) for i in range(0, len(sents), 128)])
    pairs.extend(make_pairs(sents, ppos))

# Dataset and loaders
data = PairDataset(pairs)
val_size = int(len(data) * VAL_SPLIT)
train_set, val_set = random_split(data, [len(data) - val_size, val_size], generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

# Optimizer
opt = AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * len(train_loader) * EPOCHS), len(train_loader) * EPOCHS)

# Training loop
best_val = 1e9
for ep in range(EPOCHS):
    for phase, loader in zip(["Train", "Val"], [train_loader, val_loader]):
        student.train(phase == "Train")
        total, correct, loss_sum = 0, 0, 0.0
        for a_enc, b_enc, lbl in loader:
            with torch.autocast("cuda", dtype=dtype, enabled=False):  # disable AMP to avoid dtype mismatch
                la = student(**a_enc)
                lb = student(**b_enc)
                loss = grpo_loss(la, lb)
            if phase == "Train":
                loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                opt.step(); opt.zero_grad(); sched.step()
            total += len(lbl); correct += (la > lb).sum().item(); loss_sum += loss.item() * len(lbl)
        acc = correct / total; avg_loss = loss_sum / total
        print(f"{phase} Epoch {ep+1} | loss={avg_loss:.4f} acc={acc:.3f}")
        if phase == "Val" and avg_loss < best_val:
            best_val = avg_loss
            torch.save(student.state_dict(), os.path.join(OUT_DIR, "student_best.pt"))

# Correlation check
print("\n✅ Saved best model")
student.eval()
all_scores, all_labels = [], []
for i, (a_enc, _, _) in enumerate(val_loader):
    with torch.inference_mode():
        text = student_tokenizer.batch_decode(a_enc['input_ids'], skip_special_tokens=True)
        scores = student(**a_enc).detach().cpu().numpy()
        labels = teacher_p_pos(text)
        all_scores.extend(scores); all_labels.extend(labels)
    if i > 10: break
print("Val Spearman corr (student vs teacher):", spearmanr(all_scores, all_labels).correlation)
