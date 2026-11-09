"""Audit first-day Songdo sources and prepare split-neutral observed inputs."""

import fcntl
import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_songdo_prefix_linkage as links
from scripts import fetch_m3w_songdo_first_day as fetch

pack, audit, np = links.pack, fetch.audit, links.np
PUBLIC, PRIVATE = fetch.PUBLIC, fetch.PRIVATE
MAP = audit.PUBLIC.parent / 'songdo_publisher_footprints_v1/readout.json'
OLD_META = audit.PUBLIC / 'analysis.json'
OLD_PACK = pack.PUBLIC / 'receipt.json'


def read(path):
    return json.loads(path.read_text())


def heartbeat(phase, **kw):
    p = PRIVATE / 'heartbeat.json'
    tmp = p.with_suffix('.pending')
    tmp.write_text(json.dumps(dict(pid=os.getpid(), phase=phase, **kw))+'\n')
    tmp.replace(p)
    print(dict(phase=phase, **kw), flush=True)


def expected_sessions(site):
    return {f'{audit.DATE}_{site}_{phase}{i}.csv' for phase in ('AM', 'PM') for i in range(1, 6)}


def merge(sites, labelled_edges):
    groups = {s: {s} for s in sites}
    for a, b in labelled_edges:
        g = groups[a] | groups[b]
        groups.update({s: g for s in g})
    return [list(g) for g in sorted({tuple(sorted(g)) for g in groups.values()})]


def coordinates(receipt, cache, seen, box, site):
    stem = Path(receipt['name']).stem
    p, b = cache / f'{stem}_prefix.npz', cache / f'{stem}_label_index.npz'
    if (pack.sha(p), pack.sha(b)) != (receipt['prefix_sha'], receipt['label_index_sha']):
        raise ValueError(f'Frozen input or label index changed: {stem}')
    with np.load(p, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        pack.linkage(x, y, site, seen, box)


def produce(z, name, site, binding):
    stem = Path(name).stem
    p, b = PRIVATE / f'{stem}_prefix.npz', PRIVATE / f'{stem}_label_index.npz'
    r = PRIVATE / f'{stem}_session.json'
    if r.exists():
        result = read(r)
        if result['binding'] != binding:
            raise ValueError(f'Session source or audit code changed: {stem}')
        return result
    structural = audit.session(audit.load(z, name), name, site)
    x, y = pack.arrays(z, structural)
    result = dict(binding=binding, meta=structural, prefix=dict(name=name, site=site, queries=len(x['offsets'])-1,
        agents=len(x['history']), complete_agent_futures=int(y['future_mask'].all(1).sum()),
        prefix_sha=pack.save_npz(p, x), label_index_sha=pack.save_npz(b, y)))
    pack.write_json(r, result)
    return result


def run():
    archive_path = PUBLIC / 'archives.json'
    source = read(archive_path)
    if (source['publisher_sha256'] != pack.sha(PUBLIC / 'publisher.json')
            or len(source['files']) != 20 or source['trajectory_csv_parsed'] or source['split_assigned']):
        raise ValueError('Need the verified archive acquisition receipt')
    existing, prior_inputs, maps = read(OLD_META), read(OLD_PACK), read(MAP)
    if (existing['forecast_outcomes_scored'] or prior_inputs['forecasts_scored'] or maps['track_rows_read']
            or prior_inputs['future_coordinate_fields_parsed'] or existing['new_split_assigned']
            or prior_inputs['split_assigned'] or maps['split_assigned']):
        raise ValueError('Require split-neutral source artifacts')
    impl = dict(archive_manifest=pack.sha(archive_path), implementation=pack.sha(Path(__file__)),
                parser=pack.sha(Path(pack.__file__)), clock=pack.sha(Path(audit.__file__)),
                prior_metadata=pack.sha(OLD_META), prior_input_pack=pack.sha(OLD_PACK), map_screen=pack.sha(MAP))
    old_sessions = {s['session']: s for s in existing['sessions']}
    prefix, meta, boxes, seen, member_ids = {}, [], {}, defaultdict(set), defaultdict(list)
    sources = {}
    for file, r in sorted(source['files'].items()):
        site = Path(file).stem[-1]
        archive = audit.LOCAL / file
        if pack.sha(archive) != r['sha256'] or archive.stat().st_size != r['bytes']:
            raise ValueError(f'Changed verified archive: {file}')
        reuse = site in audit.PUBLISHER
        if reuse and existing['archives'][file]['sha256'] != r['sha256']:
            raise ValueError(f'Prior source no longer matches: {file}')
        with audit.zipfile.ZipFile(archive) as z:
            members = [x for x in z.infolist() if not x.is_dir()]
            if len(members) != 10 or {x.filename for x in members} != expected_sessions(site):
                raise ValueError(f'Changed flight roster: {file}')
            for m in sorted(members, key=lambda x: x.filename):
                name = m.filename
                member_ids[(m.CRC, m.file_size)].append((site, name))
                if reuse:
                    s = old_sessions[name]
                    q = dict(prior_inputs['sessions'][name], name=name)
                    if q['binding'] != prior_inputs['binding'] or q['binding']['input_receipt'] != impl['prior_metadata']:
                        raise ValueError('Prior prefix receipt does not bind prior metadata')
                else:
                    result = produce(z, name, site, impl)
                    s, q = result['meta'], result['prefix']
                coordinates(q, pack.CACHE if reuse else PRIVATE, seen, boxes, site)
                meta.append(s)
                prefix[name] = {k: q[k] for k in ('site', 'queries', 'agents', 'complete_agent_futures',
                    'prefix_sha', 'label_index_sha')}
                prefix[name]['cache_role'] = 'previous' if reuse else 'expanded'
                heartbeat('source', completed=len(meta), session=name, queries=q['queries'])
        sources[site] = 'cached_verified' if reuse else 'fresh_run'
    site_summaries = {}
    for s in sources:
        flights = [x for x in meta if x['site'] == s]
        streams = [x for y in flights for x in y['drones']]
        stats = audit.sum_fields(x['full_visibility'] for x in streams if 'full_visibility' in x)
        p = [x for x in prefix.values() if x['site'] == s]
        if sum(x['queries'] for x in p) != stats.get('nonoverlapping_queries', 0):
            raise ValueError(f'Expanded prefix count disagrees with metadata: {s}')
        site_summaries[s] = dict(source=sources[s], rows=sum(x['rows'] for x in flights),
            sessions=len(flights), fragments=len(streams), clock_fragments=sum(x['clock_supported'] for x in streams),
            eligible_fragments=sum('full_visibility' in x for x in streams),
            sampled_agents=sum(x['agents'] for x in p),
            complete_sampled_agents=sum(x['complete_agent_futures'] for x in p), presence=stats)
    pack.write_json(PUBLIC / 'input_receipt.json', dict(binding=impl, sites=site_summaries, sessions=prefix,
        observed_steps=audit.PAST, future_steps=audit.FUTURE, nominal_frame_stride=audit.STRIDE,
        future_coordinates_parsed=False, forecasts_scored=False, split_assigned=False,
        independent_confirmation=False, reserved_outcomes_opened=False))
    edges = defaultdict(set)
    for a, b in maps['map_cutout_overlap_edges']:
        edges[(a, b)].add('map_cutout')
    envelope = []
    for a, b in combinations(sorted(boxes), 2):
        x, y = boxes[a], boxes[b]
        if np.all(np.minimum(x[1], y[1]) >= np.maximum(x[0], y[0])):
            edges[(a, b)].add('observed_envelope')
            envelope.append((a, b))
    same_drone = audit.overlap([r for s in meta for r in s['drones']])
    names = {x['session']: x['site'] for x in meta}
    for row in same_drone:
        edges[tuple(sorted(names[s] for s in row['sessions']))].add('same_drone_overlapping_clock')
    for keys in member_ids.values():
        for a, b in combinations(sorted({x[0] for x in keys}), 2):
            edges[(a, b)].add('member_crc_and_size')
    exact = 0
    for sites in seen.values():
        if len(sites) > 1:
            exact += 1
            for a, b in combinations(sorted(sites), 2):
                edges[(a, b)].add('exact_observed_prefix')
    compact_receipt = dict(sites=sources, sessions=prefix)
    close = {}
    for a, b in envelope:
        def load(site):
            return links.load(compact_receipt, pack.CACHE if site in audit.PUBLISHER else PRIVATE, site)
        match = links.compare(*load(a), *load(b))
        close[f'{a}:{b}'] = match
        if match['possible_duplicate_prefix_pairs']:
            edges[(a, b)].add('near_synchronous_observed_prefix')
        heartbeat('linkage', pair=[a, b], matches=match['possible_duplicate_prefix_pairs'])
    groups = merge(sources, edges)
    pack.write_json(PUBLIC / 'lineage.json', dict(
        input_receipt_sha256=pack.sha(PUBLIC / 'input_receipt.json'),
        candidate_source_components=groups, edges=[dict(sites=k, reasons=sorted(v)) for k, v in sorted(edges.items())],
        same_drone_overlap=same_drone, exact_cross_site_prefix_hashes=exact, near_prefix_checks=close,
        recorded_date=audit.DATE, raw_tracks_across_disjoint_times_ruled_out=False,
        clock_rule_unchanged=True, forecast_outcomes_scored=False, future_coordinates_parsed=False,
        split_assigned=False, independent_confirmation=False))
    heartbeat('complete', source_components=len(groups), sites=len(sources), sessions=len(meta))


if __name__ == '__main__':
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
