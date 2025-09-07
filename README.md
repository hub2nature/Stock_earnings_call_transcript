# Financial Sentiment Analysis Pipeline

This project contains a 3-stage sentiment scoring pipeline for earnings call transcripts. Inst]pired by https://github.com/personal-coding/Stock-Earnings-Call-Transcript-Natural-Language-Processing We have done the below experiments.

## 1. NLP-based Scoring (`NLP II`)
- Keyword-based sentiment using Loughran-McDonald dictionary.
- Backtest Results:
  - **Long-only:** Return: 4.36x | Long Hit Rate: 59.1%
  - **Long-short:** Return: 0.98x | Long: 59.1% | Short: 42.0%

## 2. FinBERT-based Scoring
- Uses `yiyanghkust/finbert-tone` model.
- Captures sentence-level financial tone (positive/neutral/negative).
- Backtest Results:
  - **Long-only:** Return: 1.42x | Long Hit Rate: 53.1%
  - **Long-short:** Return: 0.92x | Long: 53.1% | Short: 38.0%

## 3. GRPO Distillation (Optional)
- FinBERT = teacher, distilled model = student (no RL or second LLM).
- Scripts:
  - `grpo_rlaif_train.py` (train)
  - `grpo_rlaif_infer.py` (inference)
- Backtest Results:
  - **Long-only:** Return: 2.3x | Long Hit Rate: 54.0%
  - **Long-short:** Return: 0.84x | Long: 53.4% | Short: 40.0%

## Run Order

```bash
python "NLP II - Sentiment Score.py"
python "NLP II - Backtest.py"

python sentiment_score.py

python grpo_rlaif_train.py      # Optional
python grpo_rlaif_infer.py      # Optional
```

## 📁 Files
- `NLP II - Sentiment Score.py` – Rule-based scoring
- `NLP II - Backtest.py` – Strategy returns
- `sentiment_score.py` – FinBERT LLM scoring
- `grpo_rlaif_train.py` – Distilled training
- `grpo_rlaif_infer.py` – Distilled inference
