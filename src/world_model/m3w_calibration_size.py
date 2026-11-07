"""Matched calibration-size ablation with two disjoint three-source controls."""
from src.world_model.m3w_release_guard import choose_threshold


def roles(sources,held):
    names = sorted(sources)
    if len(names)!=14 or held not in names:
        raise ValueError('Require the frozen 14-source roster')
    i = names.index(held)
    first = tuple(names[(i+j)%14] for j in (1,2,3))
    second = tuple(names[(i+j)%14] for j in (4,5,6))
    fit = tuple(s for s in names if s not in (held,*first,*second))
    return fit,first,second


def comparisons(first,second):
    if len(first)!=len(second):
        raise ValueError('The same frozen grid is required')
    first_gain,second_gain,both = [],[],[]
    for x,y in zip(first,second):
        gain = None if x['lower_pp'] is None or y['lower_pp'] is None else (x['lower_pp']+y['lower_pp'])/2
        first_gain.append(dict(preserved=x['preserved'],lower_pp=gain))
        second_gain.append(dict(preserved=y['preserved'],lower_pp=gain))
        both.append(dict(preserved=x['preserved'] and y['preserved'],lower_pp=gain))
    arms = (first,second,first_gain,second_gain,both)
    return [choose_threshold(a) for a in arms],both


def mean_pair(a,b):
    return {k:(a[k]+b[k])/2 for k in ('reference','known_delta','unknown_radius')}
