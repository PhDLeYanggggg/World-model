"""Local-only contact sheets for the already decoded past-frame audit crops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image,ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=args.source.resolve()
    if not source.is_relative_to(root/'data/stage_cvpr2027_experiments'):
        raise ValueError('Third-party imagery must remain in ignored local data')
    rows=json.loads((source/'local_frame_records.json').read_text())
    output=source/'contact_sheets'
    output.mkdir(exist_ok=False)
    for scene in sorted({r['recording_id'] for r in rows}):
        ids=sorted({r['agent_id'] for r in rows if r['recording_id']==scene})
        lookup={(r['agent_id'],r['history_endpoint']):r for r in rows if r['recording_id']==scene}
        for page,start in enumerate(range(0,len(ids),6)):
            subset=ids[start:start+6]
            sheet=Image.new('RGB',(680,50+len(subset)*235),'white')
            draw=ImageDraw.Draw(sheet)
            draw.text((10,10),f'{scene}: first past / current; nearest enlargement, not body labels',fill='black')
            for i,agent in enumerate(subset):
                for j,label in enumerate(('first_past','current')):
                    r=lookup[(agent,label)]
                    if r['crop_file'] is None:
                        continue
                    with Image.open(source/scene/r['crop_file']) as im:
                        im=im.convert('RGB'); im=im.resize((im.width*2,im.height*2),Image.Resampling.NEAREST)
                    x,y=20+j*330,50+i*235
                    sheet.paste(im,(x,y))
                    draw.text((x,y+195),f"agent {agent}, {label}, index {r['requested_frame']}",fill='black')
            name=output/f'{scene}_{page}.png'
            sheet.save(name)
            print(name)


if __name__=='__main__':
    main()
