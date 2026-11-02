"""Render paper-size figures from complete public evidence, without fitting."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/evidence_manuscript_v9'
BUILD = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v9/paper_build'
EVIDENCE = ROOT/'outputs/publication_readiness_2026_09/european_selection_harm_deviance_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    summary = json.loads((EVIDENCE/'summary.json').read_text())
    verified = json.loads((EVIDENCE/'verification.json').read_text())
    if verified['summary_sha256'] != sha(EVIDENCE/'summary.json') or not verified['all54_metric_groups_exact']:
        raise ValueError('Complete exact result required')
    plt.rcParams.update({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'m3w_working_paper_v9',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(6.875,2.5))
    arms = ('mean_absolute','easy_deviance','dual_deviance'); colors = ('#5c6673','#2c8476','#b74b4d')
    for i,(a,color) in enumerate(zip(arms,colors)):
        r = summary['accuracy'][a]; value = r['point']; lo,hi = r['CI95']
        axes[0].errorbar(i,value,yerr=[[value-lo],[hi-value]],fmt='o',color=color,capsize=3)
    axes[0].set_xticks([0,1,2],['MSE','Easy','Dual']); axes[0].set_xlim(-.5,2.5)
    axes[0].set_ylabel('ADE gain over floor (%)'); axes[0].set_title('Available-label utility')
    axes[0].axhline(0,color='#666666',linestyle=':',linewidth=.7)
    for i,(a,color) in enumerate(zip(arms[1:],colors[1:])):
        r = summary['effects'][a]['completion_lower_pp']; value = r['point']; lo,hi = r['CI95']
        axes[1].errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color=color,capsize=3)
    axes[1].set_yticks([0,1],['Easy','Dual']); axes[1].set_ylim(-.5,1.5)
    axes[1].set_xlabel('Paired lower effect vs MSE (pp)'); axes[1].set_title('Registered primary: dual')
    axes[1].axvline(0,color='#666666',linestyle=':',linewidth=.7)
    for ax in axes: ax.grid(color='#dddddd',linewidth=.4); ax.set_axisbelow(True)
    fig.text(.5,.045,'Full risk support: MSE 6/252; easy 33/252; dual 30/252. No arm passes.',ha='center',fontsize=7)
    fig.subplots_adjust(left=.09,right=.99,top=.84,bottom=.26,wspace=.35)
    (PUBLIC/'figures').mkdir(parents=True,exist_ok=True); BUILD.mkdir(parents=True,exist_ok=True)
    svg = PUBLIC/'figures/cost_loss.svg'; pdf = BUILD/'cost_loss.pdf'; png = BUILD/'cost_loss.png'
    fig.savefig(svg,metadata={'Date':None}); fig.savefig(pdf,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(png,dpi=190); plt.close(fig)
    receipt = dict(result_source='fresh_plot_of_cached_verified_public_scalar_evidence',
        evidence_summary_sha256=sha(EVIDENCE/'summary.json'),script_sha256=sha(__file__),
        public_svg_sha256=sha(svg),private_PDF_sha256=sha(pdf),private_preview_sha256=sha(png),
        no_private_data_or_weights_read=True,scientific_confirmation=False,submission_ready=False)
    (PUBLIC/'figures/figure_sources.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
