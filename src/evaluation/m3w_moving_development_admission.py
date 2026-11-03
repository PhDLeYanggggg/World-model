"""Fit-only MovingDroneCrowd admission; not permission to open held sources."""
import hashlib
import json
from pathlib import Path

from src.evaluation.m3w_recording_lineage import sha256
from src.evaluation.m3w_source_reservations import REGISTRY,SourceReservations

KIND='m3w_movingdronecrowd_development_fit_admission_v1'


def eligible_train_nodes(train_nodes,components):
    if len(train_nodes)!=len(set(train_nodes)):raise ValueError('Repeated publisher train node')
    where={}
    for index,group in enumerate(components):
        for node in group:
            if node in where:raise ValueError('Source components do not partition recordings')
            where[node]=(index,group)
    if not set(train_nodes)<=where.keys():raise ValueError('Missing source grouping evidence')
    kept={};excluded={}
    for node in sorted(train_nodes):
        index,group=where[node]
        if any(v.startswith(('dut_','drone_')) for v in group):excluded[node]=group
        else:kept[node]=index
    return kept,excluded


def group_folds(records,salt):
    support={}
    for row in records.values():
        group=row['exclusion_group'];support[group]=support.get(group,0)+row['complete_native8_12_windows']
    if len(support)<3:raise ValueError('At least three conservative development groups required')
    totals=[0,0,0];assignment={}
    for group in sorted(support,key=lambda g:(-support[g],hashlib.sha256((salt+'|'+g).encode()).hexdigest())):
        fold=min(range(3),key=lambda k:(totals[k],k));assignment[group]=fold;totals[fold]+=support[group]
    return assignment,totals


def load_admission(root,path):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('Admission outside project')
    value=json.loads(path.read_text())
    if (value.get('kind')!=KIND or value.get('status')!='admitted_for_development_fit_only'
            or value.get('allowed_scientific_roles')!=['fit']
            or value.get('independent_confirmation_claimed') is not False):
        raise ValueError('Not a fit-only source admission')
    for name,digest in value['bindings'].items():
        bound=(root/name).resolve()
        if not bound.is_relative_to(root) or not bound.is_file() or sha256(bound)!=digest:
            raise ValueError('Changed source-admission evidence: '+name)
    SourceReservations(root)
    if sha256(root/REGISTRY)!=value['source_reservation_sha256']:raise ValueError('Changed protected reservations')
    if not value.get('recordings'):raise ValueError('No admitted fitting recordings')
    return value


def require_fit_record(admission,node,role,raw_path,*,held_groups=()):
    if role!='fit':raise ValueError('New source admission is fit-only, not calibration or confirmation')
    row=admission['recordings'].get(node)
    if row is None:raise ValueError('Recording is not admitted for fitting')
    if row['exclusion_group'] in held_groups:raise ValueError('Held source group entered producer fitting')
    if sha256(raw_path)!=row['source_annotation_sha256']:raise ValueError('Changed admitted annotation bytes')
    return row
