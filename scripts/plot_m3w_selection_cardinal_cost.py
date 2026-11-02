"""Deterministic figures from public scalar evidence,not training or new scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HOME=ROOT/"outputs/publication_readiness_2026_09/european_selection_cardinal_cost_v1"


def main():
    summary=json.loads((HOME/"summary.json").read_text())
    traces=json.loads((HOME/"training_traces.json").read_text())["models"]
    control=json.loads((HOME.parent/"european_selection_controller_v1/training_traces.json").read_text())["models"]
    control=[dict(v,arm="weighted_centered") for v in control if v["arm"]=="context"]
    names=("weighted_centered","mean_centered","weighted_absolute","mean_absolute")
    colors=("#555555","#237a99","#b46f29","#9c4059")
    out=HOME/"figures";out.mkdir(exist_ok=True)
    private=ROOT/"data/stage_cvpr2027_experiments/european_selection_cardinal_cost_v1/figure_preview";private.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none","svg.hashsalt":"m3w-cardinal-cost-v1",
        "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for ax,arm,color in zip(axes.flat,names,colors):
        for row in control+traces:
            if row["arm"]==arm:
                ax.plot([v["step"] for v in row["trace"]],[v["total"] for v in row["trace"]],color=color,alpha=.45,lw=.8)
        ax.set(xlabel="Actual optimizer update",ylabel="SOURCE minibatch objective",yscale="log",title=arm.replace("_"," "))
    fig.suptitle("Nine real fits per objective; loss definitions differ across panels",fontsize=12)
    fig.savefig(out/"training_traces.svg",metadata={"Date":None});fig.savefig(private/"training_traces.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout="constrained")
    for i,arm in enumerate(names[1:]):
        for offset,key,marker in ((.12,"known_pp","o"),(-.12,"completion_lower_pp","s")):
            value=summary["effects"][arm][key];point=value["point"];lo,hi=value["CI95"]
            axes[0].errorbar(point,i+offset,xerr=[[point-lo],[hi-point]],fmt=marker,color=colors[i+1],capsize=3,
                label=key.replace("_pp","").replace("_"," ") if i==0 else None)
    axes[0].axvline(0,color="#777777",lw=.8,ls="--")
    axes[0].set(yticks=range(3),yticklabels=[n.replace("_"," ") for n in names[1:]],
        xlabel="New loss minus original policy (pp)",title="Paired locality bootstrap; primary is lower")
    axes[0].legend(frameon=False,fontsize=8)
    values=[summary["safety"][n]["worst_known_easy_net_degradation_percent"] for n in names]
    axes[1].barh(range(4),values,color=colors,height=.6)
    axes[1].axvline(2,color="#333333",lw=1,ls="--",label="Original 2% easy-net requirement")
    axes[1].set(yticks=range(4),yticklabels=[n.replace("_"," ") for n in names],xlabel="Worst known easy net degradation (%)",
        title="Mean utility cannot replace all/easy positive-harm checks")
    axes[1].legend(frameon=False,fontsize=8,loc="lower right")
    fig.savefig(out/"utility_and_easy_risk.svg",metadata={"Date":None});fig.savefig(private/"utility_and_easy_risk.png",dpi=160);plt.close(fig)
    for path in out.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    print(json.dumps(dict(figures=2,raw_data_or_weights_read=False,new_training=False)))


if __name__=="__main__":main()
