"""Interpret publisher scene-or-clip lists without assigning new research roles."""
import re


def expand_publisher_list(text,clip_keys):
    clips=set(clip_keys)
    scenes={key.split('/')[0] for key in clips}
    out=[];entries=[]
    for line in text.splitlines():
        value=line.strip()
        if not value:continue
        if not re.fullmatch(r'scene_[1-9][0-9]*(?:/[1-9][0-9]*)?',value):
            raise ValueError('Unexpected publisher scene/clip identifier')
        if '/' in value:
            if value not in clips:raise ValueError('Publisher clip absent from the source inventory')
            expanded=[value];kind='clip'
        else:
            if value not in scenes:raise ValueError('Publisher scene absent from the source inventory')
            expanded=sorted(key for key in clips if key.split('/')[0]==value);kind='scene'
        entries.append(dict(identifier=value,kind=kind,expanded_clips=expanded))
        out.extend(expanded)
    if len(set(out))!=len(out):raise ValueError('Publisher list expands to duplicate clips')
    return dict(entries=entries,clips=sorted(out),scene_labels=sorted({v.split('/')[0] for v in out}))


def compare_publisher_splits(splits):
    if set(splits)!={'train','val','test'}:raise ValueError('All three publisher split lists required')
    keys={role:set(value['clips']) for role,value in splits.items()}
    scenes={role:set(value['scene_labels']) for role,value in splits.items()}
    pairs=(('train','val'),('train','test'),('val','test'))
    return dict(clip_overlaps={a+'_'+b:sorted(keys[a]&keys[b]) for a,b in pairs},
        scene_label_overlaps={a+'_'+b:sorted(scenes[a]&scenes[b]) for a,b in pairs},
        independent_physical_sites_verified=False,roles_assigned=False)
