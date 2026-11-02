"""Derived scientific figures from public scalar results and actual loss traces."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/"outputs/publication_readiness_2026_09/european_selection_controller_v1"


def main():
    traces = json.loads((HOME/"training_traces.json").read_text())
    summary = json.loads((HOME/"summary.json").read_text())
    out = HOME/"figures"; out.mkdir(exist_ok=True)
    private = ROOT/"data/stage_cvpr2027_experiments/european_selection_controller_v1/figure_preview"
    private.mkdir(exist_ok=True)
    colors = dict(row="#24758c",context="#bc4b42")
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none","svg.hashsalt":"m3w-selection-controller-v1",
                         "axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(11,3.8),layout="constrained")
    for row in traces["models"]:
        x=[r["step"] for r in row["trace"]]; y=[r["total"] for r in row["trace"]]
        label=row["arm"] if row["trial"]=="single0_seed17" else None
        axes[0].plot(x,y,color=colors[row["arm"]],alpha=.5,lw=.8,label=label)
    axes[0].set(xlabel="Actual optimizer update",ylabel="SOURCE minibatch objective",yscale="log",title="All 18 actual training traces")
    axes[0].legend(frameon=False)
    effects=[summary["primary"]["known_pp"],summary["primary"]["completion_lower_pp"]]
    points=np.asarray([r["point"] for r in effects]); bounds=np.asarray([r["CI95"] for r in effects])
    axes[1].errorbar(points,[1,0],xerr=np.stack((points-bounds[:,0],bounds[:,1]-points)),fmt="o",capsize=4,color="#333333")
    axes[1].axvline(0,color="#777777",ls="--",lw=.8)
    axes[1].set(yticks=[1,0],yticklabels=["Known secondary","Registered lower primary"],
                xlabel="Matched-joint minus independent (pp)",title="Selection-locality paired bootstrap")
    axes[1].set_ylim(-.6,1.6)
    fig.savefig(out/"training_and_primary.svg",metadata={"Date":None}); fig.savefig(private/"training_and_primary.png",dpi=160); plt.close(fig)
    names=["floor","raw_history","row_independent","row_joint_matched","context_independent","context_joint_matched","context_joint_free"]
    fig,ax=plt.subplots(figsize=(9,4.6),layout="constrained")
    values=np.asarray([summary["accuracy"][k]["point"] for k in names])
    bounds=np.asarray([summary["accuracy"][k]["CI95"] for k in names])
    labels=[k.replace("_"," ") for k in names]
    ax.errorbar(values,np.arange(len(names)),xerr=np.stack((values-bounds[:,0],bounds[:,1]-values)),fmt="o",capsize=3,color="#24758c")
    ax.axvline(0,color="#777777",ls="--",lw=.8)
    ax.set(yticks=np.arange(len(names)),yticklabels=labels,xlabel="Known available-label ADE gain vs floor (%)",
           title="Average utility does not certify record-level risk")
    ax.invert_yaxis();fig.savefig(out/"known_utility.svg",metadata={"Date":None});fig.savefig(private/"known_utility.png",dpi=160);plt.close(fig)
    for path in out.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    print(json.dumps(dict(result_source="cached_verified_scalar_visualization_not_new_experiment",figures=2,
                         raw_data_or_weights_read=False,submission_ready=False)))


if __name__=="__main__":main()
