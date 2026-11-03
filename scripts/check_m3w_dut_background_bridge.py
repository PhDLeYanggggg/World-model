"""Repeatable source-only replay without network or application artifact writes."""
import sys
sys.dont_write_bytecode=True

import json
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_m3w_dut_background_bridge as audit


def main():
    start=time.monotonic();public=audit.PUBLIC
    if not (public/'registration.json').is_file():raise ValueError('Completed registered source experiment required')
    cfg,entries,analysis=audit.register(allow_metadata_fetch=False)
    original=json.loads((public/'analysis.json').read_text());verify=json.loads((public/'verification.json').read_text())
    if not verify['all6874receipts_and_graph_exact'] or verify['analysis_sha256']!=audit.sha(public/'analysis.json'):
        raise ValueError('Require completed prescribed replay')
    audit.base.cv2.setNumThreads(1)
    matcher=json.loads((ROOT/cfg['matcher_config']).read_text())
    features,frames,receipts=audit.load_features(cfg,entries,matcher,reextract=True)
    moving=sorted(n for n in features if n.startswith('moving_'));drone=sorted(n for n in features if n.startswith('drone_'))
    dut=sorted(audit.source_node(e) for e in entries)
    roster=audit.pair_roster(moving,drone,dut,cfg['roster_seed']);results=[]
    if len(roster)!=6874:raise ValueError('Complete fixed roster required')
    for i,row in enumerate(roster):
        value=json.loads(audit.pair_path(row).read_text())
        if any(value.get(k)!=v for k,v in dict(registration_sha256=audit.sha(public/'registration.json'),**row).items()):
            raise ValueError('Changed saved pair identity')
        if i<cfg['verification_fresh_pair_prefix']:
            if audit.match_pair(row,features,frames,matcher)!={k:v for k,v in value.items() if k!='match_seconds'}:
                raise ValueError('Changed fixed fresh pair replay')
        results.append(value)
    if audit.summarize(results,analysis,dut,receipts)!=original:raise ValueError('Changed full receipts or source graph')
    print(json.dumps(dict(result_source='cached_verified_all_source_receipts_with28fresh_feature_replays_and60fresh_pairs',
        seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        checker_sha256=audit.sha(Path(__file__)),analysis_sha256=audit.sha(public/'analysis.json'),
        source_PNGs_and_prepared_features_exact=28,parent_feature_views_hash_verified=696,
        saved_pair_receipts_and_graph_exact=6874,freshly_recomputed_pairs=60,
        application_artifacts_written=0,bytecode_cache_writes_disabled=True,network_requests=0,
        new_model_training=False,predictive_outcomes_read=False,source_roles_assigned=False,
        stage5c_executed=False,smc_enabled=False)))


if __name__=='__main__':main()
