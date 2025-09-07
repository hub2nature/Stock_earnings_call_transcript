# filename: sentiment_score_llm_fast.py

import os, re, sys, codecs, time, math, traceback, warnings
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ----------------------------- CONFIG -----------------------------
IN_CSV   = "./earnings_to_search.csv"                    # expects columns: article, ticker, date
OUT_TSV  = "./LLM_loughran_mcdonald_score.txt"           # base file: word_count, pos_count, neg_count
OUT_FINAL= "./LLM_loughran_mcdonald_score_final.txt"     # adds pos_score, net_pos_score

MODEL_NAME   = "yiyanghkust/finbert-tone"
MAX_LENGTH   = 16            # shorter → faster. 128 is plenty for sentence-level.
PREP_REMARKS = True           # mimic prepared-remarks gating from original repo
DEV_LIMIT    = None           # e.g., 200 for a quick dry run
SENT_CAP     = None           # e.g., 3000 to cap sentences per transcript

# Auto batch-size: tuned for popular Colab GPUs
def suggest_batch_size(gpu_name: str) -> int:
    name = gpu_name.lower()
    if "a100" in name or "h100" in name:    # big
        return 256
    if "l4" in name:                        # mid
        return 128
    if "t4" in name or "v100" in name:      # colab standard
        return 64
    return 32                                # CPU or unknown

# ----------------------------- STARTUP -----------------------------
print("🔧 Environment check...")
has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if has_gpu else "cpu")
gpu_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
print("CUDA available:", has_gpu)
print("Device:", gpu_name)

# Matmul/cudnn speedups
if has_gpu:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

if not os.path.exists(IN_CSV):
    raise FileNotFoundError(f"Missing {IN_CSV}")

# Read index CSV, normalize column names (case-insensitive)
df_idx = pd.read_csv(IN_CSV)
lower_map = {c.lower(): c for c in df_idx.columns}
for need in ("article", "ticker", "date"):
    if need not in lower_map:
        raise ValueError(f"{IN_CSV} must contain column: {need}")
col_article = lower_map["article"]
col_ticker  = lower_map["ticker"]
col_date    = lower_map["date"]

# ----------------------------- MODEL -----------------------------
print("📦 Loading model:", MODEL_NAME)
# Choose dtype: bf16 for A100/H100, fp16 for most others, fp32 on CPU
if has_gpu:
    if any(x in gpu_name for x in ["A100", "H100"]):
        dtype = torch.bfloat16
        print("Precision: bfloat16")
    else:
        dtype = torch.float16
        print("Precision: float16")
else:
    dtype = torch.float32
    print("Precision: float32 (CPU)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=dtype)
model.eval().to(device)

# Label index mapping
id2label = model.config.id2label
label2id = {lbl.lower(): idx for idx, lbl in id2label.items()}
POS_IDX = label2id.get("positive")
NEG_IDX = label2id.get("negative")
if POS_IDX is None or NEG_IDX is None:
    # Fallback to common FinBERT order [neutral, positive, negative]
    POS_IDX = 1
    NEG_IDX = 2

BATCH_SIZE = suggest_batch_size(gpu_name)
print(f"Batch size = {BATCH_SIZE} (auto) | Max length = {MAX_LENGTH}")

# ----------------------------- HELPERS -----------------------------
WORD_RE = re.compile(r"\b[a-zA-Z']+\b")
TICKER_PAREN_RE = re.compile(r"\(([^)]+)\)")

def write_header(out_path):
    with codecs.open(out_path, "w", "utf-8") as f:
        f.write("article_link\tticker\tdate\tticker_from_text\tword_count\tpos_count\tneg_count\n")

def append_row(out_path, row):
    with codecs.open(out_path, "a", "utf-8") as f:
        f.write(row)

def read_file(file_name):
    """
    Reads transcript lines, extracts ticker from first non-'start' header line containing '(TICKER)'.
    Returns: list[list[str]](line->sentences), ticker_from_text
    """
    data, ticker = [], ""
    with codecs.open(file_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not ticker and "start" not in line.lower():
                m = TICKER_PAREN_RE.search(line)
                if m:
                    t = m.group(1)
                    if ":" in t:  # e.g., NYSE:IBM
                        t = t.split(":")[-1]
                    ticker = t
            pieces = [s.strip().lower() for s in re.split(r"\.\s+", line) if s.strip()]
            if pieces:
                data.append(pieces)
    return data, ticker

def prepared_remarks_only(flat_sentences):
    """
    Begin after basic greetings/host cues; stop at 'question-and-answer session'.
    Fallback to full text if markers missing.
    """
    start = False
    out = []
    for s in flat_sentences:
        if not start and (
            s.startswith("unknown speaker")
            or s.startswith("operator")
            or "good" in s or "hello" in s or "thank" in s or "welcome" in s
        ):
            start = True
        if s.strip() == "question-and-answer session":
            break
        if start:
            out.append(s)
    return out if out else flat_sentences

@torch.inference_mode()
def infer_probs(sent_batch):
    enc = tokenizer(
        sent_batch,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    return probs

def sentiment_counts(sentences, batch_size=BATCH_SIZE, sent_cap=SENT_CAP):
    """Word-weighted positive/negative probability aggregation."""
    if sent_cap is not None and len(sentences) > sent_cap:
        sentences = sentences[:sent_cap]

    total_words = 0
    pos_sum = 0.0
    neg_sum = 0.0

    i = 0
    bs = batch_size
    while i < len(sentences):
        batch = sentences[i:i+bs]
        try:
            probs = infer_probs(batch)  # [B, 3]
            probs = probs.float().cpu().numpy()
            for s, p in zip(batch, probs):
                wc = len(WORD_RE.findall(s))
                total_words += wc
                pos_sum += float(p[POS_IDX]) * wc
                neg_sum += float(p[NEG_IDX]) * wc
            i += bs
        except RuntimeError as e:
            # Handle CUDA OOM by reducing batch size
            if "CUDA out of memory" in str(e) and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"\n⚠️  OOM: reducing batch size to {bs}")
            else:
                raise
    return int(total_words), float(pos_sum), float(neg_sum)

# ----------------------------- AUDIT + MICRO BENCH -----------------------------
print("\n🗂️  Auditing transcripts...")
found, missing = 0, 0
for _, r in df_idx.iterrows():
    p = f".{r[col_article]}.txt"
    found += os.path.exists(p)
    missing += (not os.path.exists(p))
print(f"Found: {found} | Missing: {missing} (of {len(df_idx)})")

def micro_benchmark(n_transcripts=3, n_sentences=1200):
    paths = []
    for _, r in df_idx.iterrows():
        p = f".{r[col_article]}.txt"
        if os.path.exists(p):
            paths.append(p)
        if len(paths) >= n_transcripts:
            break
    if not paths: 
        return
    sents = []
    for p in paths:
        data, _ = read_file(p)
        flat = [s for block in data for s in block]
        sents.extend(prepared_remarks_only(flat) if PREP_REMARKS else flat)
        if len(sents) >= n_sentences:
            break
    sents = sents[:n_sentences]
    if not sents: 
        return
    t0 = time.perf_counter()
    _ = sentiment_counts(sents, batch_size=BATCH_SIZE)
    dt = time.perf_counter() - t0
    print(f"⏱️  Micro-benchmark: {len(sents)} sentences in {dt:.2f}s → {len(sents)/max(dt,1e-6):.1f} sent/s")

print("\nRunning micro-benchmark...")
micro_benchmark()

# ----------------------------- MAIN PASS -----------------------------
print("\n🚀 Scoring transcripts...")
write_header(OUT_TSV)

errors = 0
skipped = 0
processed = 0

rows = list(df_idx.itertuples(index=False))
if DEV_LIMIT: rows = rows[:DEV_LIMIT]
for r in tqdm(rows, total=len(rows)):
    article_link = getattr(r, col_article)
    ticker       = getattr(r, col_ticker)
    date         = getattr(r, col_date)

    txt_path = f".{article_link}.txt"
    if not os.path.exists(txt_path):
        continue

    try:
        text_data, ticker_from_text = read_file(txt_path)
        if not ticker_from_text:
            ticker_from_text = str(ticker)

        flat = [s for block in text_data for s in block]
        if not flat:
            skipped += 1
            continue

        sents = prepared_remarks_only(flat) if PREP_REMARKS else flat
        wc, pos, neg = sentiment_counts(sents, batch_size=BATCH_SIZE)
        append_row(
            OUT_TSV,
            f"{article_link}\t{ticker}\t{date}\t{ticker_from_text}\t{wc}\t{pos}\t{neg}\n"
        )
        processed += 1

    except Exception as e:
        errors += 1
        print(f"\n[ERROR] {article_link}: {e}")
        traceback.print_exc(limit=1, file=sys.stdout)

print(f"\n✅ Done. Processed={processed} | Empty={skipped} | Errors={errors}")
print(f"Base file: {OUT_TSV if os.path.exists(OUT_TSV) else 'NOT CREATED'}")

# ----------------------------- FINALIZE -----------------------------
if os.path.exists(OUT_TSV):
    df = pd.read_csv(OUT_TSV, sep="\t")
    denom = df["word_count"].replace(0, np.nan)
    df["pos_score"] = df["pos_count"] / denom
    df["net_pos_score"] = (df["pos_count"] - df["neg_count"]) / denom
    df = df.fillna(0)
    df.to_csv(OUT_FINAL, sep="\t", index=False)
    print(f"🎯 Final file: {OUT_FINAL}")

    # quick sanity summary
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print("\n----- Summary -----")
        print("Rows:", len(df))
        print("Unique tickers:", df["ticker_from_text"].nunique())
        print("Date range:", df["date"].min(), "→", df["date"].max())
        print("Head:\n", df[["article_link","ticker","date","ticker_from_text","word_count","pos_score","net_pos_score"]].head().to_string(index=False))
        print("\nnet_pos_score stats:\n", df["net_pos_score"].describe().to_string())
    except Exception as e:
        print("Summary failed:", e)
else:
    print("No output to finalize.")
