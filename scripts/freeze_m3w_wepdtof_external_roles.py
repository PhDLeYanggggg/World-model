"""Freeze the approved WEPDTOF source roles without opening outcome positions."""
import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_source_probe as probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cafile',default='/etc/ssl/cert.pem')
    a = parser.parse_args()
    p = probe.OUT
    paths = [p/k for k in ('registration_v3.json','support.json','pilot_readout.json',
                           'rapid_comparison.json','input_contrast.json')]
    g,s,m,r,c = [json.loads(q.read_text()) for q in paths]
    if (s['registration_sha256']!=probe.sha(paths[0]) or m['registration_sha256']!=probe.sha(paths[0])
            or c['parent_sha256']!=probe.sha(paths[3]) or any(z['independent_confirmation'] for z in (g,s,m,r,c))
            or s['nonpilot_model_scored'] or not r['support_gate']['rapid']):
        raise ValueError('Require structural-only nonpilot data and a scored engineering pilot')
    groups = {}
    for row in s['clips']:
        k = row['source']
        if probe.data.source(row['clip']) != k:
            raise ValueError('Clip regrouping detected')
        g0 = groups.setdefault(k,dict(clips=[],role='development' if k==probe.PILOT else 'frozen_external_once',
                                     structural_multi_agent_stride_four=False))
        g0['clips'].append(row['clip'])
        g0['structural_multi_agent_stride_four'] |= row['support']['4']['queries']>0
    if len(groups)!=14 or sum(len(x['clips']) for x in groups.values())!=16:
        raise ValueError('Unexpected source groups')
    with probe.archive(a.cafile) as archive, ZipFile(archive) as z:
        probe.check_version(archive)
        catalog = probe.data.catalog(z.namelist())
    if set(catalog)!={clip for row in groups.values() for clip in row['clips']}:
        raise ValueError('Archive differs from source support receipt')
    result = dict(approval='user selected one-shot external verification, 2026-11-09',
        code_sha256=probe.sha(Path(__file__)), archive_version=g['archive_version'],
        parents={q.name:probe.sha(q) for q in paths}, groups=groups, development_sources=1,
        external_sources=13, external_structural_multi_agent_sources=sum(
            r['structural_multi_agent_stride_four'] for r in groups.values() if r['role']!='development'),
        roles_frozen=True, selection_on_external=False, external_labels_opened_for_scoring=False,
        next_required='freeze external image-only predictions and exact risk/utility readout before scoring',
        original_2pct=True, independent_confirmation=False, deployment_changed=False,
        dronecrowd_calibration_groups=15, sealed_dronecrowd_confirmation_groups=30)
    probe.immutable(p/'external_roles.json',result)
    print({k:result[k] for k in ('development_sources','external_sources','external_structural_multi_agent_sources',
                                 'external_labels_opened_for_scoring')})


if __name__=='__main__':
    main()
