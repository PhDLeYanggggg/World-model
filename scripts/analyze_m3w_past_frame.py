"""Separate static-history frame diagnostics from forecasting accuracy."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 environment')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_objective_alignment import supported_standardization
from src.world_model.m3w_observed_motion import feature_variant
from src.world_model.m3w_past_frame import past_frame, rotate_features, frame_prediction


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    folder = ROOT/'outputs/publication_readiness_2026_09/past_frame'
    report = json.loads((folder/'report.json').read_text())
    replay = json.loads((folder/'replay.json').read_text())
    if len(replay['trials']) != 54 or replay['identity'] != report['identity'] or not all(t['prediction_exact'] for t in replay['trials']):
        raise ValueError('Full exact replay required')
    source = ROOT/'data/stage_cvpr2027_experiments/offline_visual_forecast/inputs'
    motion_path = ROOT/'data/stage_cvpr2027_experiments/observed_motion_v2/inputs'
    identity = report['identity']['source_identity']
    if file_digest(source/'data_manifest.json') != identity['source_manifest_sha256'] or file_digest(motion_path/'manifest.json') != identity['motion_manifest_sha256']:
        raise ValueError('Source identity changed')
    manifest = json.loads((source/'data_manifest.json').read_text())
    a = {}
    for n in ('geometry','baselines','folds','image_rows','coverage'):
        if file_digest(source/(n+'.npy')) != manifest['arrays'][n+'.npy']:
            raise ValueError('Changed source array')
        a[n] = np.load(source/(n+'.npy'),mmap_mode='r')
    mm = json.loads((motion_path/'manifest.json').read_text())
    for n in ('motion.npy','quality.npy'):
        if file_digest(motion_path/n) != mm['arrays'][n]:
            raise ValueError('Changed motion array')
    motion,quality = [np.load(motion_path/(n+'.npy'),mmap_mode='r') for n in ('motion','quality')]
    observed = torch.cat([torch.from_numpy(a['coverage'][a['image_rows'][i:i+128],None].astype(np.float32)/9.)
                          .mean((2,3,4)) for i in range(0,len(a['folds']),128)])
    cv = manifest['baseline_names'].index('constant_velocity_causal_fd')
    base = a['baselines'][:,cv]
    static = np.all(a['geometry'][:,:16] == 0,axis=1)
    matrices = (np.array([[0.,-1.],[1.,0.]]),-np.eye(2),np.array([[0.,1.],[-1.,0.]]))
    diagnostics = {}
    for trial in report['trials']:
        cp,pp = ROOT/trial['checkpoint_path'],ROOT/trial['prediction_path']
        if file_digest(cp) != trial['checkpoint_sha256'] or file_digest(pp) != trial['prediction_sha256']:
            raise ValueError('Frozen model changed')
        train = np.flatnonzero(a['folds'] != trial['fold'])
        held = np.flatnonzero(a['folds'] == trial['fold'])
        ids = held[static[held]]
        if not len(ids):
            diagnostics[trial['trial']] = dict(static_rows=0,status='no_static_support')
            continue
        original = feature_variant(a['geometry'],motion,quality,trial['variant'])
        q,valid,_ = past_frame(original)
        features = rotate_features(original,q).astype(np.float32) if trial['arm'] == 'past_frame' else original
        _,normalizer = supported_standardization(features[train],features)
        torch.manual_seed(trial['seed']); model = OfflineVisualForecast(581)
        model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']); model.eval()
        with np.load(pp) as saved:
            if not np.array_equal(saved['held_indices'],held):
                raise ValueError('Row alignment changed')
            original_pred = saved['prediction'][static[held]]
        gaps = []
        for matrix in matrices:
            r = np.broadcast_to(matrix,(len(ids),2,2))
            reexpressed = rotate_features(original[ids],r)
            qr,vr,_ = past_frame(reexpressed)
            xx = (rotate_features(reexpressed,qr) if trial['arm'] == 'past_frame' else reexpressed).astype(np.float32)
            z = np.clip((xx-normalizer['mean'])/normalizer['std'],-10,10)
            z[:,normalizer['constant']] = 0
            if trial['arm'] != 'past_frame':
                qr = np.broadcast_to(np.eye(2),(len(ids),2,2))
            if trial['arm'] == 'original':
                vr = np.ones(len(ids),bool)
            bb = np.einsum('ntd,ndk->ntk',base[ids],r).astype(np.float32)
            with torch.no_grad():
                predicted = frame_prediction(model,torch.from_numpy(z.astype(np.float32)),observed[ids],
                    torch.from_numpy(bb),torch.from_numpy(qr.astype(np.float32)),torch.from_numpy(vr)).numpy()
            gaps.append(np.einsum('ntd,nkd->ntk',predicted,r)-original_pred)
        gap = np.asarray(gaps)
        diagnostics[trial['trial']] = dict(static_rows=len(ids),mean_gap=float(np.linalg.norm(gap,axis=-1).mean()),
            max_coordinate_gap=float(np.abs(gap).max()),unsupported_rows=int((~valid[ids]).sum()),labels_used=False)
    conditions = {}
    for name,s in report['summary'].items():
        chosen = [t for t in report['trials'] if t['variant']+'_'+t['arm'] == name]
        ds = [diagnostics[t['trial']] for t in chosen if diagnostics[t['trial']]['static_rows']]
        easy = [x for x in s['easy_degradation_percent'] if x is not None]
        conditions[name] = dict(s,static_query_rotation_gap_mean=float(np.mean([d['mean_gap'] for d in ds])),
            static_query_rotation_gap_max=max(d['max_coordinate_gap'] for d in ds),
            easy_degradation_range_percent=[min(easy),max(easy)],
            static_stay_ADE=[t['slices']['static_stays'].get('primary_ADE') for t in chosen],
            static_move_gain_percent=[t['slices']['static_moves'].get('improvement_percent') for t in chosen])
    result = dict(report_sha256=file_digest(folder/'report.json'),replay_sha256=file_digest(folder/'replay.json'),
        analysis_code_sha256=file_digest(Path(__file__)),result_source='fresh_run_target_free_static_frame_check_cached_verified_models',
        conditions=conditions,per_fit_static_diagnostics=diagnostics,
        interpretation='Moving ego source pipeline already heading-aligns. Rotating its aligned inputs is not evidence of a raw-pipeline bug. Static-only gaps are reported separately.',
        fresh_fit_seconds=sum(t['fit']['fit_seconds'] for t in report['trials'] if t['arm'] != 'original'),
        new_model_updates=0,primary_changed=False,sealed_roles_opened=False,deployment=False)
    json_write(folder/'analysis.json',result)
    private = ROOT/'data/stage_cvpr2027_experiments/past_frame'; cache = private/'plot_runtime'; cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache); os.environ['XDG_CACHE_HOME'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'m3w-past-frame-v1'})
    arms = ('original','guard_only','past_frame')
    fig,axes = plt.subplots(1,2,figsize=(12,4.8))
    for shift,v,color in [(-.18,'quality_control','#3977a8'),(.18,'directed','#b96435')]:
        axes[0].bar(np.arange(3)+shift,[conditions[v+'_'+a]['vs_CV']['gain_percent'] for a in arms],.35,label=v,color=color)
        axes[1].bar(np.arange(3)+shift,[conditions[v+'_'+a]['static_query_rotation_gap_mean'] for a in arms],.35,label=v,color=color)
    for ax in axes:
        ax.set_xticks(np.arange(3),arms,rotation=15); ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,fontsize=9)
    axes[0].set_ylabel('Primary forecasting gain vs CV (%)'); axes[0].axhline(0,color='#333333',linewidth=.7)
    axes[1].set_ylabel('Static-history coordinate gap (normalized)')
    fig.suptitle('Coordinate consistency is not forecasting improvement')
    fig.text(.5,.01,'Fixed seeds and budgets; exposed fit scenes. Quarter-turn stress uses no future labels.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(folder/'frame_comparison.svg',metadata={'Date':None}); fig.savefig(private/'frame_comparison.png',dpi=140); plt.close(fig)
    print(json.dumps({k:dict(gain=v['vs_CV']['gain_percent'],gain_vs_guard=v['vs_guard']['gain_percent'],
        static_rotation_gap=v['static_query_rotation_gap_mean'],easy_range=v['easy_degradation_range_percent'],
        safe_fits=v['safe_positive_fits']) for k,v in conditions.items()},indent=2))


if __name__ == '__main__':
    main()
