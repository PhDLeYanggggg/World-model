"""B-only event frequency and fitted-cost diagnosis before a sampler experiment."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selected_risk_learning as parent
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_easy_harm_sampling_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_easy_harm_sampling_v1'


def check_parent():
    _, identity = parent.registration()
    v = json.loads((parent.PUBLIC/'final_verification.json').read_text()); assert v['all_passed']
    for p,h in v['artifacts'].items(): assert parent.digest(parent.PUBLIC/p) == h
    for p,h in v['source_bindings'].items(): assert parent.digest(ROOT/p) == h
    parent.checked_training(identity)
    return identity


def summarize(y, prediction, use):
    value = y[use, 3]; pred = prediction[use, 3]; n = len(value)
    positive = value > 0; mass = float(value.sum())
    top = max(1, int(np.ceil(n*.01)))
    return dict(rows=n, positive_rows=int(positive.sum()), positive_rate=float(positive.mean()) if n else None,
        target_mass=mass, predicted_mass=float(pred.sum()),
        predicted_over_actual=float(pred.sum()/mass) if mass > 0 else None,
        top_one_percent_mass_share=float(np.sort(value)[-top:].sum()/mass) if mass > 0 else None,
        maximum=float(value.max()) if n else None,
        positive_median=float(np.median(value[positive])) if positive.any() else None)


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    identity = check_parent(); PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    rows = []
    for g,data,pairs in parent.contexts(identity):
        bi = pairs['B']['ids']
        for pair in ('full','motion_only'):
            path = PRIVATE/'audit'/(g['group']+'_'+pair+'.json')
            if path.exists():
                r = json.loads(path.read_text()); assert r['parent'] == identity
                rows.append(r); continue
            bx,be,by,masks,pr,*_ = parent.pair_inputs(g,data,pairs,pair)
            directory = parent.PRIVATE/'heads'/(g['group']+'_'+pair+'_mean')
            model,state = parent.restore(directory)
            pred = parent.method.predict(model,bx,be,pr)
            known = pr['known']; p = float(np.dot(pr['weights'], np.nan_to_num(by[:,3]) > 0))
            r = dict(parent=identity, group=g['group'], pair=pair, source='fresh_run_B_only_diagnosis',
                input_sha256=parent.array_hash(bx), target_sha256=parent.array_hash(by),
                model=parent.artifact(directory/'checkpoint.pt'), positive_sampling_probability=p,
                expected_positives_per_batch=256*p, zero_positive_batch_probability=(1-p)**256,
                by_site={str(s):{name:summarize(by,pred,known & (data['sites'][bi] == s) & mask)
                    for name,mask in (('population',np.ones(len(bi),bool)),('old_raw_selected',masks[:,2]))}
                    for s in sorted(set(data['sites'][bi]))}, C_labels_used=False)
            parent.immutable_json(path,r); rows.append(r)
            print(json.dumps(dict(group=g['group'],pair=pair,expected_positives=256*p)),flush=True)
    parent.immutable_json(PUBLIC/'training_support.json',dict(parent_verification=parent.artifact(parent.PUBLIC/'final_verification.json'),
        source_code_sha256=parent.digest(Path(__file__)), groups=rows, groups_count=len(rows),
        C_outcomes_used=False, new_training=False))


if __name__ == '__main__': main()
