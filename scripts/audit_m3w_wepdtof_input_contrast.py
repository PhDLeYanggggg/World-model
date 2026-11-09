"""Compare camera and annotation histories on the exact common pilot targets."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_rapid_probe as rapid
from scripts.run_m3w_acme_gmu_camera import rollout

base = rapid.base


def main():
    _, mobile = rapid.register()
    path = base.OUT/'rapid_comparison.json'
    report = rapid.load(path)
    freeze = base.OUT/'rapid_freeze.json'
    fr = rapid.load(freeze)
    if (report['registration'] != base.sha(rapid.REG) or report['freeze_sha256'] != base.sha(freeze)
            or report['independent_confirmation'] or not report['support_gate']['matched']):
        raise ValueError('Require the scored common pilot roster')
    audit = rapid.load(base.OUT/'support.json')
    pilot, = [c for c in audit['clips'] if c['clip']==base.PILOT]
    a = base.LOCAL/f'{base.PILOT}.json'
    if base.sha(a)!=pilot['annotation_sha256'] or pilot['quality']['unlisted_images']:
        raise ValueError('Pilot labels changed')
    d = rapid.load(a)
    frames = {base.data.frame_number(x['file_name'],base.PILOT):x['file_name'] for x in d['images']}
    people,_ = base.data.annotation_table(d,base.PILOT,frames,geometry=True)
    q = {k:[] for k in ('mobile','rapid','annotation_history')}
    for m,r in zip(mobile['windows'],fr['windows'],strict=True):
        s = m['start']
        p = rapid.PRIVATE/f'window_{s:06}.json'
        if r['start'] != s or base.sha(p)!=r['cache_sha256']:
            raise ValueError('Invalid frozen comparison')
        a,b = rapid.match(rapid.verify_source(m),people),rapid.match(rapid.load(p),people)
        use = sorted(set(a)&set(b))
        q['mobile'].append((s,{j:a[j] for j in use}))
        q['rapid'].append((s,{j:b[j] for j in use}))
        q['annotation_history'].append((s,{j:rollout([people[j][s+k*base.STRIDE][:2]
                                                    for k in range(8)]) for j in use}))
    result = {k:rapid.score(a,people) for k,a in q.items()}
    if any(result[k]!=report['common_scores'][k] for k in ('mobile','rapid')):
        raise ValueError('Original common-target scores did not reproduce')
    if len({tuple(sorted(z['counts'].items())) for z in result.values()})!=1:
        raise ValueError('History contrast changed target or label support')
    base.immutable(base.OUT/'input_contrast.json',dict(parent_sha256=base.sha(path),code_sha256=base.sha(Path(__file__)),
        results=result, one_camera_development_only=True, original_2pct=True,
        independent_confirmation=False, new_model_training=False))
    print(result,flush=True)


if __name__ == '__main__':
    main()
