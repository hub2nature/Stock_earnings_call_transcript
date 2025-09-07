import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

# -------- your existing params ----------
quantile_value = 0.20
portfolio_value = 1000.0
months_behind = 4
long_hit_rate = long_count = short_hit_rate = short_count = 0

# -------- load scores ----------
df = pd.read_csv('./LLM_RLgrpo_score_final.txt', sep='\t')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

topic_name = 'net_pos_score'  # already present in *_final.txt

# Optional: alias old tickers -> current CSV names (add more as needed)
TICKER_ALIAS = {
    'SYMC': 'GEN',   # Symantec → Gen Digital
    'FB': 'META',    # Facebook → Meta
    'GOOG': 'GOOGL'  # if your prices are only under one class
}

def csv_path_for(tkr: str) -> str:
    t = TICKER_ALIAS.get(tkr, tkr).replace('.', '-')
    return f'./stock_price/{t}.csv'

def next_open_after(tkr: str, date_str: str):
    path = csv_path_for(tkr)
    if not os.path.exists(path):
        return None
    px = pd.read_csv(path, parse_dates=['Date'])
    s = px[px['Date'] > pd.to_datetime(date_str)]
    if s.empty:
        return None
    return float(s.iloc[0]['Open'])

# -------- helpers for resume metrics ----------
def max_drawdown(pv):
    pv = np.asarray(pv, dtype=float)
    running_max = np.maximum.accumulate(pv)
    dd = pv / running_max - 1.0
    return float(dd.min()) if len(dd) else 0.0

def annualized_return(pv0, pvn, n_months):
    if n_months <= 0 or pv0 <= 0:
        return 0.0
    return (pvn / pv0) ** (12.0 / n_months) - 1.0

def annualized_sharpe(monthly_returns, rf=0.0):
    r = np.asarray(monthly_returns, dtype=float)
    if r.size == 0:
        return 0.0
    ex = r - rf/12.0
    mu = ex.mean()
    sig = ex.std(ddof=1)
    return float((mu / sig) * np.sqrt(12.0)) if sig > 0 else 0.0

# -------- tracking for summary ----------
pv_path = [portfolio_value]
monthly_returns = []
dates_printed = []

# -------- backtest loop ----------
for beg in pd.date_range('2012-01-01', '2022-04-30', freq='MS'):
    start_date = beg.strftime("%Y-%m-%d")
    end_date = (beg + MonthEnd(months_behind)).strftime("%Y-%m-%d")
    end_date_one_month_later = (beg + MonthEnd(months_behind + 1)).strftime("%Y-%m-%d")

    prev_pv = portfolio_value  # for this month’s return calc

    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    data_subset = df.loc[mask].drop_duplicates('ticker_from_text', keep='last')

    # pick top/bottom quantiles
    buys = data_subset[data_subset[topic_name] >= data_subset[topic_name].quantile(1 - quantile_value)]['ticker_from_text'].unique().tolist()
    shorts = data_subset[data_subset[topic_name] <= data_subset[topic_name].quantile(quantile_value)]['ticker_from_text'].unique().tolist()

    # keep only tickers that actually have a CSV and at least one future quote
    buys_filtered = []
    for t in buys:
        if os.path.exists(csv_path_for(t)) and next_open_after(t, end_date) is not None and next_open_after(t, end_date_one_month_later) is not None:
            buys_filtered.append(t)
    shorts_filtered = []
    for t in shorts:
        if os.path.exists(csv_path_for(t)) and next_open_after(t, end_date) is not None and next_open_after(t, end_date_one_month_later) is not None:
            shorts_filtered.append(t)

    nlegs = len(buys_filtered) + len(shorts_filtered)
    if nlegs == 0:
        # no trades this month → 0% monthly return
        dates_printed.append(end_date_one_month_later)
        pv_path.append(portfolio_value)
        monthly_returns.append(0.0)
        print(f'{end_date_one_month_later}\t{portfolio_value}\t{long_count}\t{long_hit_rate}\t{short_count}\t{short_hit_rate}')
        continue

    invest_in_each = portfolio_value / nlegs

    # longs
    for ticker in buys_filtered:
        start_open = next_open_after(ticker, end_date)
        end_open = next_open_after(ticker, end_date_one_month_later)
        if start_open is None or end_open is None:
            continue
        long_count += 1
        if end_open >= start_open:
            long_hit_rate += 1
        portfolio_value += invest_in_each * (end_open / start_open - 1)

    # shorts
    for ticker in shorts_filtered:
        start_open = next_open_after(ticker, end_date)
        end_open = next_open_after(ticker, end_date_one_month_later)
        if start_open is None or end_open is None:
            continue
        gain_on_short = start_open - end_open
        short_count += 1
        if start_open >= end_open:
            short_hit_rate += 1
        portfolio_value += invest_in_each * gain_on_short / start_open

    # print your original line
    print(f'{end_date_one_month_later}\t{portfolio_value}\t{long_count}\t{long_hit_rate}\t{short_count}\t{short_hit_rate}')

    # track PV & monthly return
    dates_printed.append(end_date_one_month_later)
    pv_path.append(portfolio_value)
    monthly_returns.append((portfolio_value / prev_pv) - 1.0)

# -------- final resume-friendly summary ----------
start_pv = pv_path[0]
end_pv = pv_path[-1]
n_months = len(monthly_returns)

ls_cagr   = annualized_return(start_pv, end_pv, n_months)
ls_sharpe = annualized_sharpe(monthly_returns)
ls_mdd    = max_drawdown(pv_path)

long_hr  = (long_hit_rate / long_count) if long_count > 0 else 0.0
short_hr = (short_hit_rate / short_count) if short_count > 0 else 0.0

print("\n===== Backtest Summary (resume-ready) =====")
print(f"Period: {dates_printed[0]} → {dates_printed[-1]}  |  Months: {n_months}")
print(f"Start PV: {start_pv:,.2f}  End PV: {end_pv:,.2f}  Total Return: {(end_pv/start_pv - 1):.2%}")
print(f"Long–Short CAGR: {ls_cagr:.2%}  |  Sharpe (monthly→annual): {ls_sharpe:.2f}  |  Max Drawdown: {ls_mdd:.2%}")
print(f"Long hits: {long_hit_rate}/{long_count}  ({long_hr:.1%})  |  Short hits: {short_hit_rate}/{short_count}  ({short_hr:.1%})")
print(f"Avg legs/month: {( (np.array([len(set(df[(df['date']>=pd.to_datetime(d)-MonthEnd(months_behind)) & (df['date']<=pd.to_datetime(d))]['ticker_from_text']) ) for d in dates_printed])) ).mean():.1f}  (approx.)")
print("===========================================")
