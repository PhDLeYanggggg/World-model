"""Derived seed-spread experiment figures from public scalar records only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
HOME=ROOT/"outputs/publication_readiness_2026_09/european_selection_uncertainty_v1"


def main():
    summary=json.loads((HOME/"summary.json").read_text());traces=json.loads((HOME/"training_traces.json").read_text())["models"]
    out=HOME/"figures";out.mkdir(exist_ok=True)
    private=ROOT/"data/stage_cvpr2027_experiments/european_selection_uncertainty_v1/figure_preview";private.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none","svg.hashsalt":"m3w-selection-uncertainty-v1",
        "axes.spines.top":False,"axes.spines.right":False})
    names=("disabled","real","noise");colors=("#555555","#237a99","#b35a45")
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    for ax,arm,color in zip(axes,names,colors):
        for row in traces:
            if row["arm"]==arm:ax.plot([v["step"] for v in row["trace"]],[v["total"] for v in row["trace"]],color=color,alpha=.45,lw=.8)
        ax.set(xlabel="Actual optimizer update",ylabel="SOURCE minibatch objective",yscale="log",title=arm)
    fig.suptitle("Nine actual fits per input arm; same loss and complete query draws",fontsize=12)
    fig.savefig(out/"training_traces.svg",metadata={"Date":None});fig.savefig(private/"training_traces.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for i,name in enumerate(("real_minus_disabled","real_minus_noise")):
        for offset,key,marker in ((.10,"known_pp","o"),(-.10,"completion_lower_pp","s")):
            x=summary["effects"][name][key];point=x["point"];lo,hi=x["CI95"]
            axes[0].errorbar(point,i+offset,xerr=[[point-lo],[hi-point]],fmt=marker,color=colors[i+1],capsize=3,
                label=key.replace("_pp","").replace("_"," ") if i==0 else None)
    axes[0].axvline(0,color="#777777",ls="--",lw=.8)
    axes[0].set(yticks=(0,1),yticklabels=("real minus disabled","real minus noise"),
        xlabel="Paired utility difference (pp)",title="Locality bootstrap\nSquares: completion lower; circles: known")
    axes[0].legend(frameon=False,fontsize=8)
    values=[summary["safety"][arm]["worst_known_easy_net_degradation_percent"] for arm in names]
    axes[1].barh(range(3),values,color=colors,height=.6)
    axes[1].axvline(2,color="#333333",ls="--",lw=1)
    axes[1].set(yticks=range(3),yticklabels=names,xlabel="Worst known easy net degradation (%)",
        title="Worst recording: easy net\nPositive-harm guards also apply")
    axes[1].set_ylim(-.5,2.9)
    axes[1].text(2.1,2.55,"Original 2% easy-net requirement",fontsize=8,color="#333333")
    fig.savefig(out/"utility_and_risk.svg",metadata={"Date":None});fig.savefig(private/"utility_and_risk.png",dpi=160);plt.close(fig)
    for path in out.glob("*.svg"):path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    print(json.dumps(dict(figures=2,raw_data_or_weights_read=False,new_training=False)))


if __name__=="__main__":main()
