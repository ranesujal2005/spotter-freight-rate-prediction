import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED=12000
EXPECTED_IDS={f"TE-{i:06d}" for i in range(1,EXPECTED+1)}
DATES=pd.date_range("2025-12-01","2025-12-31",freq="D")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--december-predictions",required=True)
    ap.add_argument("--output-dir",default="scorer_results")
    a=ap.parse_args()
    p=pd.read_csv(a.predictions)
    if list(p.columns)!=["load_id","predicted_rate"] or len(p)!=EXPECTED: raise ValueError("Predictions schema/row count invalid")
    if p.load_id.isna().any() or p.load_id.duplicated().any() or set(p.load_id.astype(str))!=EXPECTED_IDS: raise ValueError("Prediction IDs invalid")
    r=pd.to_numeric(p.predicted_rate,errors="coerce")
    if r.isna().any() or not np.isfinite(r).all() or (r<=0).any(): raise ValueError("Invalid predicted_rate")
    d=pd.read_csv(a.december_predictions)
    cols=["pickup","delivery","distance","equipment","weight","date","predicted_rate"]
    if list(d.columns)!=cols or len(d)!=31: raise ValueError("December schema/row count invalid")
    d["date"]=pd.to_datetime(d.date,errors="coerce")
    for c in ["distance","weight","predicted_rate"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    if d.isna().any().any() or not np.isfinite(d[["distance","weight","predicted_rate"]]).all().all(): raise ValueError("Invalid December values")
    if d.date.duplicated().any() or set(d.date)!=set(DATES): raise ValueError("December dates invalid")
    if not d.pickup.eq("Lexington").all() or not d.delivery.eq("Fort Wayne").all(): raise ValueError("Route invalid")
    if not np.isclose(d.distance,360).all() or not d.equipment.eq("Dry Van").all() or not np.isclose(d.weight,32000).all(): raise ValueError("Fixed inputs invalid")
    if (d.predicted_rate<=0).any(): raise ValueError("December predictions must be positive")
    d=d.sort_values("date")
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10.8,4.8),dpi=180)
    color="#064A56"
    ax.plot(d.date,d.predicted_rate,color=color,linewidth=2.6,marker="o",markersize=3.2)
    floor=float(d.predicted_rate.min())
    ax.fill_between(d.date,d.predicted_rate,floor-max(10.0,floor*.02),color=color,alpha=.08)
    ax.set_title("Candidate: December 2025 Predicted Load Rate",loc="left",fontsize=15,fontweight="bold",pad=12)
    ax.set_ylabel("Predicted rate ($)")
    ax.grid(axis="y",color="#D9E2E4",linewidth=.8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#9DAFB3")
    ax.tick_params(axis="x",rotation=35)
    ax.text(0,-.40,"Fixed inputs: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes",transform=ax.transAxes,fontsize=9.5,color="#455A60")
    fig.tight_layout(rect=(0,.12,1,1))
    fig.savefig(out/"candidate_december.png",bbox_inches="tight")
    plt.close(fig)
    print(f"Validated {EXPECTED:,} final predictions.")
    print("Validated 31 fixed December predictions.")
    print(f"Created chart: {out/'candidate_december.png'}")
if __name__=="__main__": main()
