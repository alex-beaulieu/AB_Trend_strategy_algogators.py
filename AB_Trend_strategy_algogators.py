"""Reproduce trend-related Summary outputs from Prices_Monthly.

Run:
  python trend_summary.py workbook.xlsx --output-dir trend_results
"""
from __future__ import annotations
import argparse, json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook

ASSETS = [
    "GSCI Gasoil", "GSCI Gold", "Japan Gov't Bond",
    "US 10Y Gov't Bond FUTURES INDEX",
    "S&P 500 Futures Excess Rtn Index",
    "Nasdaq 100 Futures Excess Rtn Index",
    "SGI FX Forwards JPY/USD in USD",
]
SP500, NASDAQ, USBOND = ASSETS[4], ASSETS[5], ASSETS[3]

@dataclass(frozen=True)
class Config:
    workbook: str
    sheet: str = "Prices_Monthly"
    start_date: str = "2000-10-31"
    end_date: str = "2026-07-31"
    signal_lookback: int = 12
    vol_lookback: int = 36
    annualization: int = 12
    leverage: float = 4.22  # Current Summary formula; header rounds to 4.23x.
    annual_cash_return: float = 0.0190923064311557
    annual_cost: float = 0.0
    missing_policy: str = "available"  # available or strict
    output_dir: str = "trend_results"

def load_prices(cfg):
    raw = pd.read_excel(cfg.workbook, sheet_name=cfg.sheet, header=None)
    names = raw.iloc[1].astype("string").str.strip()
    tickers = raw.iloc[2].astype("string").str.strip()
    meta = pd.DataFrame({"excel_column": range(1, raw.shape[1]+1),
                         "display_name": names, "ticker": tickers})
    data = raw.iloc[3:].copy()
    data.columns = ["date" if i == 0 else names.iloc[i] for i in range(raw.shape[1])]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    data = data.loc[~data.index.duplicated(keep="last")]
    missing = [x for x in ASSETS if x not in data.columns]
    if missing: raise KeyError(f"Missing Prices_Monthly series: {missing}")
    prices = data[ASSETS].apply(pd.to_numeric, errors="coerce")
    if cfg.missing_policy == "strict" and prices.loc[cfg.start_date:cfg.end_date].isna().any().any():
        raise ValueError("Missing values in requested period.")
    return prices, meta

def build_strategy(prices, cfg):
    returns = prices.pct_change(fill_method=None)
    trailing = prices.div(prices.shift(cfg.signal_lookback)).sub(1)
    signal = np.sign(trailing)
    vol = returns.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).std(ddof=1)
    vol *= np.sqrt(cfg.annualization)
    inv = 1 / vol
    proportional = inv.div(inv.sum(axis=1), axis=0)
    # The workbook carries the first allocation with complete volatility
    # histories backward over the initial partial-history period.
    complete = inv.notna().all(axis=1)
    proportional = proportional.where(complete).bfill()
    signed_weights = signal * proportional
    applied_weights = signed_weights.shift(1)  # prevents look-ahead
    valid = returns.notna() & applied_weights.notna()
    if cfg.missing_policy == "strict":
        trend = (applied_weights * returns).sum(axis=1).where(valid.all(axis=1))
    else:
        trend = (applied_weights.where(valid) * returns.where(valid)).sum(axis=1, min_count=1)
    monthly_cost = (1 + cfg.annual_cost) ** (1/cfg.annualization) - 1
    levered = trend * cfg.leverage - monthly_cost
    contributions = applied_weights.mul(returns).mul(cfg.leverage)
    trend.name, levered.name = "Trend Return", "Trend Levered Return"
    return locals()

def period(x, cfg): return x.loc[cfg.start_date:cfg.end_date].dropna()
def ann_return(r, cfg): return np.prod(1+r)**(cfg.annualization/r.count())-1+cfg.annual_cash_return
def ann_risk(r, cfg): return r.std(ddof=1)*np.sqrt(cfg.annualization)
def drawdown(r):
    wealth=(1+r).cumprod(); peak=wealth.cummax()
    return pd.DataFrame({"Return":r,"Wealth":wealth,"Peak":peak,"Drawdown":wealth/peak-1})

def capture(strategy, benchmark, up, cfg):
    z=pd.concat([strategy, benchmark], axis=1).dropna()
    mask=z.iloc[:,1].gt(0) if up else z.iloc[:,1].lt(0)
    return (z.loc[mask].iloc[:,0].mean()*12)/(z.loc[mask].iloc[:,1].mean()*12)

def performance_summary(model, cfg):
    source={"S&P 500":period(model["returns"][SP500],cfg),
            "Nasdaq 100":period(model["returns"][NASDAQ],cfg),
            "Trend Levered":period(model["levered"],cfg),
            "Trend Return":period(model["trend"],cfg)}
    out={}
    for name,r in source.items():
        risk=ann_risk(r,cfg); ar=ann_return(r,cfg)
        dd=drawdown(r)["Drawdown"]
        out[name]={"Start Date":r.index.min(),"End Date":r.index.max(),
                   "Total Return":np.prod(1+r),"Annualized Return":ar,
                   "Annualized Risk":risk,
                   "Sharpe":(ar-cfg.annual_cash_return)/risk,
                   "Maximum Drawdown":dd.min(),
                   "Hit Ratio":(r>0).sum()/311}  # exact workbook denominator
    lev,sp=source["Trend Levered"],source["S&P 500"]
    active=lev-sp.reindex(lev.index); bond=period(model["returns"][USBOND],cfg).reindex(lev.index)
    out["Trend Levered"].update({
        "Tracking Error to S&P 500":ann_risk(active,cfg),
        "Correlation to S&P 500":lev.corr(sp),
        "Correlation to US Bonds":lev.corr(bond),
        "Information Ratio":(out["Trend Levered"]["Annualized Return"]-out["S&P 500"]["Annualized Return"])/ann_risk(active,cfg),
        "Upside Capture":capture(lev,sp,True,cfg),
        "Downside Capture":capture(lev,sp,False,cfg)})
    out["Trend Levered"]["Capture Differential"]=out["Trend Levered"]["Upside Capture"]-out["Trend Levered"]["Downside Capture"]
    return pd.DataFrame(out)

def quintiles(sp, trend, levered):
    rank=sp.rank(method="min",ascending=False); pct=rank/sp.count()
    q=pd.cut(pct,[0,.2,.4,.6,.8,1],labels=["Q1","Q2","Q3","Q4","Q5"],include_lowest=True)
    z=pd.concat([sp.rename("S&P 500"),trend,levered,q.rename("Quintile")],axis=1).dropna()
    return z.groupby("Quintile",observed=False).mean(numeric_only=True)*12, pd.DataFrame({"Percentile":pct,"Quintile":q})

EVENTS=[
 ("Dot-com Bust / 2001 Recession","2000-10-31","2002-12-31"),
 ("Credit Expansion","2003-01-31","2007-11-30"),
 ("Global Financial Crisis","2007-12-31","2009-06-30"),
 ("Post-GFC / Euro Stress","2009-07-31","2012-12-31"),
 ("Low-Vol Expansion","2013-01-31","2019-12-31"),
 ("COVID Shock","2020-01-31","2020-04-30"),
 ("Reopening & Inflation Surge","2020-05-31","2022-02-28"),
 ("Rapid Tightening","2022-03-31","2024-08-31"),
 ("Easing / Late Cycle","2024-09-30","2026-07-31")]
MACRO=[
 ("Post-2000 Easing","2000-10-31","2004-06-30"),
 ("2004-07 Hiking Cycle","2004-07-31","2007-07-31"),
 ("GFC Easing / ZIRP Transition","2007-08-31","2008-12-31"),
 ("ZIRP / QE Low Inflation","2009-01-31","2015-11-30"),
 ("Gradual Tightening","2015-12-31","2019-07-31"),
 ("Pre-COVID Easing + Pandemic ZIRP","2019-08-31","2022-02-28"),
 ("Inflation Surge + Rapid Hiking","2022-03-31","2024-08-31"),
 ("Disinflation + Easing","2024-09-30","2026-07-31")]

def regime_table(levered, contrib, regimes):
    rows=[]
    for label,start,end in regimes:
        r=levered.loc[start:end].dropna(); c=contrib.loc[start:end].sum()
        rows.append({"Regime":label,"Start":start,"End":end,"Months":r.count(),
                     "Levered Trend Return":np.prod(1+r)-1,
                     **{f"{k} Contribution":v for k,v in c.items()}})
    return pd.DataFrame(rows).set_index("Regime")

def build_outputs(cfg):
    prices,meta=load_prices(cfg); m=build_strategy(prices,cfg)
    sp=period(m["returns"][SP500],cfg); tr=period(m["trend"],cfg); lv=period(m["levered"],cfg)
    qsum,qrows=quintiles(sp,tr,lv)
    monthly=pd.concat([sp.rename("SP500 Return"),period(m["returns"][NASDAQ],cfg).rename("Nasdaq Return"),tr,lv,qrows],axis=1)
    return {
      "Monthly Summary":monthly, "Performance Summary":performance_summary(m,cfg),
      "Quintile Summary":qsum, "Event Attribution":regime_table(lv,m["contributions"],EVENTS),
      "Macro Attribution":regime_table(lv,m["contributions"],MACRO),
      "Sleeve Contributions":m["contributions"].loc[cfg.start_date:cfg.end_date],
      "Drawdowns":pd.concat([drawdown(tr).add_prefix("Trend "),drawdown(lv).add_prefix("Levered ")],axis=1),
      "Weights":m["applied_weights"].loc[cfg.start_date:cfg.end_date], "Metadata":meta}

def validate_against_workbook(outputs,cfg,monthly_tol=1e-8,summary_tol=5e-4):
    """Compare Python results with cached Excel values without recalculating Excel."""
    wb=load_workbook(cfg.workbook,data_only=True,read_only=True)
    ws=wb["Summary"]
    py=outputs["Monthly Summary"]
    checks=[]
    for row in range(2,312):
        date=pd.Timestamp(ws.cell(row,1).value)
        if date not in py.index: continue
        for cell,col in ((f"H{row}","Trend Return"),(f"I{row}","Trend Levered Return")):
            excel=ws[cell].value; python=float(py.loc[date,col])
            diff=np.nan if not isinstance(excel,(int,float)) else python-float(excel)
            checks.append({"Item":cell,"Excel":excel,"Python":python,"Difference":diff,
                           "Tolerance":monthly_tol,"Pass":isinstance(excel,(int,float)) and abs(diff)<=monthly_tol})
    ps=outputs["Performance Summary"]
    mapping={
      "X42":("Trend Levered","Total Return"),"X43":("Trend Levered","Annualized Return"),
      "X44":("Trend Levered","Annualized Risk"),"X45":("Trend Levered","Sharpe"),
      "X46":("Trend Levered","Tracking Error to S&P 500"),
      "X47":("Trend Levered","Correlation to S&P 500"),
      "X48":("Trend Levered","Correlation to US Bonds"),"X49":("Trend Levered","Hit Ratio"),
      "X51":("Trend Levered","Information Ratio"),"X52":("Trend Levered","Upside Capture"),
      "X53":("Trend Levered","Downside Capture"),"X54":("Trend Levered","Capture Differential"),
      "Y42":("Trend Return","Total Return"),"Y43":("Trend Return","Annualized Return"),
      "Y44":("Trend Return","Annualized Risk")}
    for cell,(series,metric) in mapping.items():
        excel=ws[cell].value; python=float(ps.loc[metric,series])
        diff=np.nan if not isinstance(excel,(int,float)) else python-float(excel)
        checks.append({"Item":cell,"Excel":excel,"Python":python,"Difference":diff,
                       "Tolerance":summary_tol,"Pass":isinstance(excel,(int,float)) and abs(diff)<=summary_tol})
    return pd.DataFrame(checks)

def export(outputs,cfg):
    out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True)
    with pd.ExcelWriter(out/"trend_summary_python.xlsx",engine="openpyxl") as w:
        for name,df in outputs.items(): df.to_excel(w,sheet_name=name[:31])
    for name,df in outputs.items(): df.to_csv(out/f"{name.lower().replace(' ','_')}.csv")
    (out/"configuration.json").write_text(json.dumps(asdict(cfg),indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument("workbook"); p.add_argument("--output-dir",default="trend_results")
    p.add_argument("--start-date",default="2000-10-31");p.add_argument("--end-date",default="2026-07-31")
    p.add_argument("--leverage",type=float,default=4.22);p.add_argument("--signal-lookback",type=int,default=12)
    p.add_argument("--vol-lookback",type=int,default=36);p.add_argument("--cash-return",type=float,default=.0190923064311557)
    p.add_argument("--annual-cost",type=float,default=0);p.add_argument("--missing-policy",choices=["available","strict"],default="available")
    a=p.parse_args(); cfg=Config(a.workbook,start_date=a.start_date,end_date=a.end_date,leverage=a.leverage,
      signal_lookback=a.signal_lookback,vol_lookback=a.vol_lookback,annual_cash_return=a.cash_return,
      annual_cost=a.annual_cost,missing_policy=a.missing_policy,output_dir=a.output_dir)
    o=build_outputs(cfg)
    o["Parity Check"]=validate_against_workbook(o,cfg)
    export(o,cfg)
    print(o["Performance Summary"].round(6))
    print("\\nParity:",o["Parity Check"]["Pass"].value_counts(dropna=False).to_dict())
if __name__=="__main__": main()

