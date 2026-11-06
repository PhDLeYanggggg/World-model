"""Private past-frame alignment sheets; no raw source image is published."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_past_background as run
from PIL import Image, ImageDraw


def main():
    reg = run.read(run.REG); run.read(run.PUBLIC/'pilot.json')
    home = run.PRIVATE/'figures'; home.mkdir(parents=True, exist_ok=True)
    records = reg['roster']['records']; references = []
    for start in range(0, len(records), 12):
        group = records[start:start+12]; canvas = Image.new('RGB', (3*480, ((len(group)+2)//3)*305), 'white')
        for j, record in enumerate(group):
            q = record['query_frames'][0]; cache = run.raster(record['node'], q)
            rgb = Image.fromarray(cache['gray']).convert('RGB'); draw = ImageDraw.Draw(rgb)
            indexed = run.annotation_index(record['annotation']['path'], record['annotation']['sha256'])
            boxes = indexed.get(q, run.np.empty((0, 4)))*run.np.tile(cache['resize_scale'], 2)
            for x0, y0, x1, y1 in boxes:
                draw.rectangle((x0, y0, x1, y1), outline='#00FFFF', width=1)
            rgb.thumbnail((470, 270)); x, y = (j%3)*480, (j//3)*305
            canvas.paste(rgb, (x, y+24))
            measured = run.read(run.PRIVATE/'pairs'/record['node']/f'{q:06d}.json')['measurement']
            ImageDraw.Draw(canvas).text((x+3, y+3), f'{record["node"]} f{q} supported={measured["supported"]}', fill='black')
            references.append(dict(node=record['node'], annotation_frame=q, image_frame=q+1,
                raster=run.artifact(run.PRIVATE/'rasters'/record['node']/f'{q:06d}.npz')))
        canvas.save(home/f'pilot_alignment_{start//12+1}.png')
    run.atomic(home/'alignment_sources.json', dict(views=references, future_frame_used=False,
        review_scope='first_query_current_frame_in_each_of34fitting_recordings_not_all_images'))
    print('Three private fixed34view sheets rendered; not predictive or metric evidence.')


if __name__ == '__main__':
    main()
