"""Private source-review contact sheets, never model inputs or public raw media."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts import audit_m3w_movingdronecrowd_background as parent_audit


def font():
    path=Path('/System/Library/Fonts/Menlo.ttc')
    return ImageFont.truetype(str(path),12) if path.exists() else ImageFont.load_default()


def save_contact(canvas,path,resume=False):
    if path.exists():
        if not resume:
            raise ValueError('Preserve previous source-review media')
        with Image.open(path) as existing:
            if existing.mode!=canvas.mode or not np.array_equal(np.asarray(existing),np.asarray(canvas)):
                raise ValueError('Changed source-review pixels during resume')
    else:
        canvas.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',action='store_true');parser.add_argument('--version2',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    audit=parent_audit
    if args.version2:
        from scripts import audit_m3w_movingdronecrowd_background_v2 as audit
    public,private=audit.PUBLIC,audit.PRIVATE
    manifest=json.loads((public/'image_manifest.json').read_text())
    if manifest['registration_sha256']!=audit.sha(public/'registration.json'):
        raise ValueError('Changed source images')
    records={(v['node'],v['slot']):v for v in manifest['views']}
    output=private/'visual_review';output.mkdir(exist_ok=True)
    receipt=public/('pair_contact_sources.json' if args.pairs else 'overview_contact_sources.json')
    if receipt.exists():raise ValueError('Completed figure source manifest is immutable')
    outputs=[];typeface=font()
    old_lookup=None
    def view(node,slot):
        nonlocal old_lookup
        if node.startswith('moving_'):
            row=records[node,slot];path=ROOT/row['path']
            if audit.sha(path)!=row['sha256']:raise ValueError('Changed prepared source view')
            with np.load(path,allow_pickle=False) as arrays:image=Image.fromarray(arrays['gray'].copy()).convert('RGB')
            return image,row['image_frame_one_based'],row['source_JPEG_sha256']
        if old_lookup is None:
            oldmanifest=ROOT/'outputs/publication_readiness_2026_09/dronecrowd_image_audit_v1/source_manifest.json'
            identity=json.loads((audit.OLD_PRIVATE/'identity.json').read_text())
            if audit.sha(oldmanifest)!=identity['image_manifest_sha256']:raise ValueError('Changed protected image source')
            old_lookup={(v['scene_id'],v['frame_one_based']):v for v in json.loads(oldmanifest.read_text())['frames']}
        frame=(1,150,300)[slot-1];row=old_lookup[node.removeprefix('drone_'),frame]
        path=ROOT/'external_data/DroneCrowd_images'/row['local_name']
        if audit.sha(path)!=row['sha256']:raise ValueError('Changed protected image')
        with Image.open(path) as raw:image=raw.convert('RGB')
        return image,frame,row['sha256']
    def save(canvas,name,sources):
        path=output/name
        save_contact(canvas,path,args.resume)
        outputs.append(dict(path=str(path.relative_to(ROOT)),sha256=audit.sha(path),sources=sources))
    if not args.pairs:
        nodes=sorted({node for node,slot in records if slot==1})
        for page in range(0,len(nodes),12):
            canvas=Image.new('RGB',(960,508),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,5),'MovingDroneCrowd++ first views: source audit, not forecasting',font=typeface,fill='black')
            sources=[]
            for cell,node in enumerate(nodes[page:page+12]):
                image,frame,digest=view(node,1);x=(cell%4)*240;y=28+(cell//4)*160
                canvas.paste(image.resize((240,135)),(x,y))
                draw.text((x+3,y+138),node.removeprefix('moving_')+f' f{frame}',font=typeface,fill='black')
                sources.append(dict(node=node,slot=1,actual_image_frame=frame,source_JPEG_sha256=digest))
            save(canvas,f'overview_{page//12+1:02d}.png',sources)
        selection_rule='all120first_views_sorted_by_namespaced_clip'
    else:
        summary=json.loads((public/'analysis.json').read_text())
        cases=[]
        for status in ('strong_multiview','ambiguous'):
            candidates=sorted((v for v in summary['supported_pairs'] if v['kind']=='cross_source' and v['status']==status),
                              key=lambda v:(v['left'],v['right']))
            cases.extend(candidates[:8])
        for index,case in enumerate(cases,1):
            canvas=Image.new('RGB',(960,630),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,5),case['status']+' candidate; no geographic certification',font=typeface,fill='black')
            sources=[]
            supports=case['support'][:2]
            if case['status']=='strong_multiview':
                strong=[v for v in case['support'] if v['status']=='strong']
                supports=next(pair for pair in itertools.combinations(strong,2)
                              if pair[0]['left_frame']!=pair[1]['left_frame']
                              and pair[0]['right_frame']!=pair[1]['right_frame'])
            for row,support in enumerate(supports):
                for column,side in enumerate(('left','right')):
                    node=case[side];slot=support[side+'_frame'];image,frame,digest=view(node,slot)
                    x=column*480;y=28+row*300
                    canvas.paste(image.resize((480,270)),(x,y))
                    draw.text((x+4,y+273),node+f' actual f{frame}',font=typeface,fill='black')
                    sources.append(dict(node=node,slot=slot,actual_image_frame=frame,source_JPEG_sha256=digest))
            save(canvas,f'cross_candidate_{index:02d}.png',sources)
        selection_rule='first8lexical_cross_pairs_per_strong_or_ambiguous_status_not_all_pairs'
    parent_audit.once(receipt,dict(renderer_sha256=audit.sha(Path(__file__)),image_manifest_sha256=audit.sha(public/'image_manifest.json'),
        selection_rule=selection_rule,outputs=outputs,visual_review_performed=False,
        figures_private_not_committed=True,model_inputs=False,physical_independence_certified=False))
    print(json.dumps(dict(contact_sheets=len(outputs),manifest=str(receipt))))


if __name__=='__main__':main()
