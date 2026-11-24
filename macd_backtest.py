"""159516 MACD 金叉/死叉回测（今年以来）

规则：MACD 柱 (DIF-DEA)*2 由负转正（由绿翻红）当日收盘买入（满仓），
      由正转负（由红转绿）当日收盘卖出（空仓）。不计手续费，不做空。
参数组：(12,26,9) (6,13,5) (8,17,6)
指标用 2025-06 起的数据预热，交易从今年第一个交易日开始，期初空仓。

用法：
  python macd_backtest.py                # 通过 akshare 拉取数据（需能访问东方财富）
  python macd_backtest.py data.csv       # 使用本地 CSV（需含 date, close 列）
"""
import sys
from datetime import date

import pandas as pd

CODE = "159516"
PARAMS = [(12, 26, 9), (6, 13, 5), (8, 17, 6)]
YEAR_START = pd.Timestamp(date.today().year, 1, 1)
WARMUP_START = "20250601"


def load_data():
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
    else:
        import akshare as ak
        df = ak.fund_etf_hist_em(symbol=CODE, period="daily", start_date=WARMUP_START,
                                 end_date=date.today().strftime("%Y%m%d"), adjust="qfq")
        df = df.rename(columns={"日期": "date", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)


def macd_hist(close, fast, slow, sig):
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=sig, adjust=False).mean()
    return 2 * (dif - dea)


def backtest(df, fast, slow, sig):
    hist = macd_hist(df["close"], fast, slow, sig)
    d = df.assign(h=hist)
    d["prev"] = d["h"].shift(1)
    d = d[d["date"] >= YEAR_START].reset_index(drop=True)

    cash, shares, entry = 1.0, 0.0, None
    trades, equity = [], []
    for _, r in d.iterrows():
        if shares == 0 and r.prev <= 0 < r.h:          # 绿翻红 -> 买
            shares, cash, entry = cash / r.close, 0.0, r
        elif shares > 0 and r.prev >= 0 > r.h:         # 红转绿 -> 卖
            cash, shares = shares * r.close, 0.0
            trades.append((entry.date.date(), entry.close, r.date.date(), r.close, r.close / entry.close - 1))
        equity.append(cash + shares * r.close)
    if shares > 0:  # 仍持仓，按最新收盘计
        last = d.iloc[-1]
        trades.append((entry.date.date(), entry.close, "持仓中", last.close, last.close / entry.close - 1))

    eq = pd.Series(equity)
    mdd = (eq / eq.cummax() - 1).min()
    wins = sum(t[4] > 0 for t in trades)
    return {
        "参数": f"{fast},{slow},{sig}",
        "总收益": eq.iloc[-1] - 1,
        "交易次数": len(trades),
        "胜率": wins / len(trades) if trades else float("nan"),
        "最大回撤": mdd,
    }, trades, d


def main():
    df = load_data()
    rows = []
    for p in PARAMS:
        res, trades, d = backtest(df, *p)
        rows.append(res)
        print(f"\n=== MACD({res['参数']}) 交易明细 ===")
        for t in trades:
            print(f"  买 {t[0]} @{t[1]:.3f}  卖 {t[2]} @{t[3]:.3f}  收益 {t[4]:+.2%}")
    bh = d["close"].iloc[-1] / d["close"].iloc[0] - 1
    print(f"\n区间: {d['date'].iloc[0].date()} ~ {d['date'].iloc[-1].date()}，"
          f"买入持有收益: {bh:+.2%}\n")
    out = pd.DataFrame(rows)
    for c in ["总收益", "胜率", "最大回撤"]:
        out[c] = out[c].map(lambda x: f"{x:+.2%}" if c != "胜率" else f"{x:.0%}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
