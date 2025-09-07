# filename: grpo_rlaif_infer.py
import os, re, codecs, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------- CONFIG ----------------
IN_CSV    = "./earnings_to_search.csv"
OUT_TSV   = "./LLM_RLgrpo_score.txt"
OUT_FINAL = "./LLM_RLgrpo_score_final.txt"

STUDENT_BACKBONE = "distilbert-base-uncased"
CKPT_PATH  = "./grpo_student_ckpt/student_best.pt"

MAX_LENGTH = 32
PREP_REMARKS = True
SENT_CAP     = None       # Set to int for sentence limit
DEV_LIMIT    = 1000         # Set None to disable row cap

# ---------------- UTILS ----------------
WORD_RE = re.compile(r"\b[a-zA-Z']+\b")
TICKER_PAREN_RE = re.compile(r"\(([^)]+)\)")

def read_file(file_name):
    data, ticker = [], ""
    if not os.path.exists(file_name): return [], ""
    with codecs.open(file_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if not ticker and "start" not in line.lower():
                m = TICKER_PAREN_RE.search(line)
                if m: ticker = m.group(1).split(":")[-1]
            data.append([s.strip().lower() for s in re.split(r"\.\s+", line) if s.strip()])
    return data, ticker

def prepared_remarks_only(flat):
    start, out = False, []
    for s in flat:
        s_low = s.lower()
        if not start and (s_low.startswith("unknown") or "good" in s_low or "hello" in s_low):
            start = True
        if s_low.strip() == "question-and-answer session": break
        if start: out.append(s)
    return out or flat

def write_header(path):
    with codecs.open(path, "w", "utf-8") as f:
        f.write("article_link\tticker\tdate\tticker_from_text\tword_count\tpos_count\tneg_count\n")

def append_row(path, row):
    with codecs.open(path, "a", "utf-8") as f:
        f.write(row)

# ---------------- STUDENT MODEL ----------------
class SentimentHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.linear(x)

class StudentModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hsz = backbone.config.hidden_size
        self.head = SentimentHead(hsz)
    def forward(self, input_ids=None, attention_mask=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)

# ---------------- SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_BACKBONE, use_fast=True)
backbone  = AutoModel.from_pretrained(STUDENT_BACKBONE).to(device)
student   = StudentModel(backbone).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)
# If using simple nn.Linear head (not .linear): rename keys if needed
if "head.weight" in ckpt:
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("head.weight"): new_ckpt["head.linear.weight"] = v
        elif k.startswith("head.bias"): new_ckpt["head.linear.bias"] = v
        else: new_ckpt[k] = v
    ckpt = new_ckpt

student.load_state_dict(ckpt)
student.eval()
print("\U0001F4E6 Loaded student model from:", CKPT_PATH)

@torch.inference_mode()
def student_p_pos(batch_text):
    enc = tokenizer(batch_text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = student(**enc)
    return torch.sigmoid(logits).cpu().numpy()

# ---------------- MAIN ----------------
assert os.path.exists(IN_CSV), f"Missing {IN_CSV}"
df_idx = pd.read_csv(IN_CSV)
colmap = {c.lower(): c for c in df_idx.columns}
col_article, col_ticker, col_date = colmap["article"], colmap["ticker"], colmap["date"]

write_header(OUT_TSV)

rows = list(df_idx.itertuples(index=False))
if DEV_LIMIT: rows = rows[:DEV_LIMIT]

for r in tqdm(rows):
    article_link = getattr(r, col_article)
    ticker = getattr(r, col_ticker)
    date = getattr(r, col_date)
    txt_path = f".{article_link}.txt"
    if not os.path.exists(txt_path): continue

    try:
        data, ticker_from_text = read_file(txt_path)
        if not ticker_from_text: ticker_from_text = ticker
        flat = [s for block in data for s in block]
        if not flat: continue
        sents = prepared_remarks_only(flat) if PREP_REMARKS else flat
        if SENT_CAP and len(sents) > SENT_CAP: sents = sents[:SENT_CAP]

        word_count = pos_sum = neg_sum = 0.0
        for i in range(0, len(sents), 256):
            batch = sents[i:i+256]
            probs = student_p_pos(batch)
            for s, p in zip(batch, probs):
                wc = len(WORD_RE.findall(s))
                word_count += wc
                pos_sum += p * wc
                neg_sum += (1 - p) * wc

        append_row(OUT_TSV, f"{article_link}\t{ticker}\t{date}\t{ticker_from_text}\t{int(word_count)}\t{pos_sum}\t{neg_sum}\n")

    except Exception as e:
        print("[ERROR]", article_link, e)

# ---------------- FINALIZE ----------------
if os.path.exists(OUT_TSV):
    df = pd.read_csv(OUT_TSV, sep="\t")
    denom = df["word_count"].replace(0, np.nan)
    df["pos_score"] = df["pos_count"] / denom
    df["net_pos_score"] = (df["pos_count"] - df["neg_count"]) / denom
    df = df.fillna(0)
    df.to_csv(OUT_FINAL, sep="\t", index=False)
    print("\u2705 Wrote:", OUT_FINAL)
else:
    print("No output produced.")
