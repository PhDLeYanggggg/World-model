"""DUT source-image evidence with honest single-background support semantics."""
import itertools

import numpy as np

from src.evaluation.m3w_dronecrowd_grouping import pair_evidence
from src.evaluation.m3w_dronecrowd_image_geometry import candidate_components


def select_backgrounds(tree,clips,expected_tree,expected_bytes):
    if tree.get('sha')!=expected_tree or tree.get('truncated'):
        raise ValueError('Require the complete pinned DUT tree')
    wanted={f'data/backgrounds/{name}_background_GSOC.png' for name in clips}
    rows=[v for v in tree['tree'] if v['path'].startswith('data/backgrounds/')]
    if {v['path'] for v in rows}!=wanted or len(rows)!=len(wanted):
        raise ValueError('Require all and only the pinned author background PNGs')
    if any(v.get('type')!='blob' or v.get('mode')!='100644'
           or not isinstance(v.get('size'),int) or not 0<v['size']<=2097152 for v in rows):
        raise ValueError('Accept only bounded regular source PNG blobs')
    if sum(v['size'] for v in rows)!=expected_bytes:
        raise ValueError('Changed exact source-background byte inventory')
    return [{k:v[k] for k in ('path','size','sha')} for v in sorted(rows,key=lambda v:v['path'])]


def pair_roster(moving,drone,dut,seed):
    nodes=list(moving)+list(drone)+list(dut)
    if len(nodes)!=len(set(nodes)):raise ValueError('Source namespaces overlap')
    rng=np.random.default_rng(seed)
    sets=[]
    for kind,values in (('moving_to_dut',itertools.product(sorted(moving),sorted(dut))),
                        ('drone_to_dut',itertools.product(sorted(drone),sorted(dut))),
                        ('within_dut',itertools.combinations(sorted(dut),2))):
        values=list(values)
        sets.append([dict(kind=kind,left=values[i][0],right=values[i][1]) for i in rng.permutation(len(values))])
    return [row for group in itertools.zip_longest(*sets) for row in group if row is not None]


def compare_available_views(left,right,config):
    if len(right)!=1:raise ValueError('One real DUT background, never repeated temporal slots')
    supported=[];attempted=0
    for a,b in itertools.product(sorted(left),sorted(right)):
        evidence=pair_evidence(left[a],right[b],config);attempted+=1
        if evidence['status']!='unsupported':supported.append(dict(left_slot=a,right_slot=b,**evidence))
    return dict(status='ambiguous' if supported else 'unsupported',support=supported,
        frame_pairs_attempted=attempted,left_available_views=len(left),right_available_views=1,
        DUT_background_is_verified_temporal_frame=False,multiview_support_claimed=False)


def conservative_bridge(parent_groups,dut_nodes,results):
    sites={}
    for node in dut_nodes:
        if node.startswith('dut_intersection_'):site='dut_intersection'
        elif node.startswith('dut_roundabout_'):site='dut_shared_space'
        else:raise ValueError('Unknown DUT source-site identity')
        sites.setdefault(site,[]).append(node)
    groups=[*parent_groups,*[sorted(v) for v in sites.values()]]
    nodes=sorted(node for group in groups for node in group)
    if len(nodes)!=len(set(nodes)):raise ValueError('Conservative source groups must partition recordings')
    edges=[(g[i-1],g[i]) for g in groups for i in range(1,len(g))]
    edges += [(r['left'],r['right']) for r in results if r['status']!='unsupported']
    if any(a not in nodes or b not in nodes for a,b in edges):raise ValueError('Unknown source node in a match')
    components=candidate_components(nodes,edges)
    linked=sorted(node for group in components if any(v.startswith(('dut_','drone_')) for v in group)
                  for node in group if node.startswith('moving_'))
    conflicts=[g for g in components if any(v.startswith('dut_') for v in g) and any(v.startswith('drone_') for v in g)]
    return dict(conservative_components=components,moving_nodes_connected_to_reserved_sources=linked,
        DUT_DroneCrowd_cross_reservation_components=conflicts,
        publisher_and_parent_and_DUT_site_constraints_preserved=True,
        negative_matching_certifies_geographic_independence=False,scientific_roles_assigned=False)
