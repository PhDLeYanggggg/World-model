"""Build a native-stride working manuscript from pinned public scalar evidence."""
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlsplit

import jinja2
import mistune

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = HOME/'evidence_manuscript_v10'
BUILD = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/paper_build'
PINNED = {
    'moving_visual_risk_v1/analysis.json': '4375e369a675ff2f396e5502a8e3bf43cb33dbce173142cb8ff2e4c8ecd7025f',
    'moving_visual_risk_v1/calibration_analysis.json': '9ca842da055628c2341502c9e442c66411118b3faacbba7c5aabf3798063551b',
    'moving_visual_risk_v1/failure_case.json': '5eb109a4a98f9f3076bfe3b95e94120b0572f66bc81360e90bce04d2a3b47523',
    'moving_prefix_actor_v1/analysis.json': 'ae140b4cad607c07883446a4f0b8c31fe8512cbe7d262ebcd1f78b329f91e83e',
    'moving_supported_global_motion_v1/analysis.json': 'ff55b3039123a2c6d8a5b8b58147e1d69aa96609bc4131c4ee050208397cfb74',
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def tex_escape(text):
    chars = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(chars.get(c, c) for c in text)


def inlines(nodes):
    out = []
    for n in nodes:
        kind = n['type']; children = n.get('children', [])
        if kind == 'text': out.append(tex_escape(n['raw']))
        elif kind in ('softbreak', 'linebreak'): out.append(' ')
        elif kind in ('strong', 'emphasis', 'codespan'):
            command = {'strong': 'textbf', 'emphasis': 'emph', 'codespan': 'texttt'}[kind]
            out.append('\\'+command+'{'+(tex_escape(n['raw']) if kind == 'codespan' else inlines(children))+'}')
        elif kind == 'inline_math': out.append('$'+n['raw']+'$')
        elif kind == 'link':
            uri = n['attrs']['url']
            if urlsplit(uri).scheme not in ('https', 'http'): raise ValueError('Paper links must be public web references')
            label = ''.join(x.get('raw', '') for x in children)
            safe_uri = uri.replace('%', r'\%').replace('#', r'\#')
            out.append(r'\url{'+safe_uri+'}' if label == uri else r'\href{'+safe_uri+'}{'+inlines(children)+'}')
        else: raise ValueError('Unsupported inline AST node: '+kind)
    return ''.join(out)


def title_text(node):
    return ''.join(x.get('raw', '') for x in node.get('children', []))


def strip_number(title):
    head, sep, tail = title.partition(' ')
    return tail if sep and all(x.isdigit() for x in head.rstrip('.').split('.')) else title


def latex_document(markdown):
    ast = mistune.create_markdown(renderer='ast', plugins=['table', 'math', 'url'])(markdown)
    title = title_text(ast[0]); body = []; abstract = False; caption = None
    for n in ast[1:]:
        kind = n['type']
        if kind == 'blank_line': continue
        if kind == 'heading':
            text = title_text(n); level = n['attrs']['level']
            if level == 3 and text.startswith('Table '): caption = text.partition('. ')[2]; continue
            if abstract: body.append(r'\end{abstract}'); abstract = False
            if text == 'Abstract': body.append(r'\begin{abstract}'); abstract = True; continue
            if text == 'References': body.append(r'\section*{References}'); continue
            if text == 'LLM Usage Statement': body.append(r'\section*{LLM Usage Statement}'); continue
            command = 'section' if level == 2 else 'subsection'
            body.append('\\'+command+'{'+tex_escape(strip_number(text))+'}')
        elif kind == 'paragraph':
            kids = n['children']
            if len(kids) == 1 and kids[0]['type'] == 'image':
                img = kids[0]; path = img['attrs']['url']
                if path not in ('decision_pipeline.pdf', 'visual_cost_evidence.pdf'): raise ValueError('Only generated scalar figures permitted')
                body.append(r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{'+path+'}'+
                    r'\caption{'+inlines(img.get('children', []))+r'}\end{figure*}')
            else: body.append(inlines(kids)+'\n')
        elif kind == 'block_math': body.append('\\[\n'+n['raw']+'\n\\]')
        elif kind == 'table':
            head, rows = n['children']; cells = head['children']; count = len(cells)
            if caption is None: raise ValueError('Every table requires an AST caption heading')
            narrow = count <= 4
            opening = r'\noindent\begin{minipage}{\columnwidth}\small' if narrow else r'\begin{table*}[t]\centering\small'
            caption_cmd = r'\captionof{table}' if narrow else r'\caption'
            width = r'\columnwidth' if narrow else r'\textwidth'
            tab = [opening, caption_cmd+'{'+tex_escape(caption)+'}',
                r'\setlength{\tabcolsep}{4pt}', r'\begin{tabularx}{'+width+'}{'+('>{\\raggedright\\arraybackslash}X'*count)+r'}\toprule',
                ' & '.join(r'\textbf{'+inlines(c['children'])+'}' for c in cells)+r' \\\midrule']
            for row in rows['children']:
                if len(row['children']) != count: raise ValueError('Unequal Markdown table width')
                tab.append(' & '.join(inlines(c['children']) for c in row['children'])+r' \\')
            closing = r'\end{minipage}\par\medskip' if narrow else r'\end{table*}'
            tab += [r'\bottomrule\end{tabularx}', closing]; body.append('\n'.join(tab)); caption = None
        elif kind == 'list':
            ordered = n.get('attrs', {}).get('ordered', False); env = 'enumerate' if ordered else 'itemize'
            body.append('\\begin{'+env+'}')
            for item in n['children']:
                text = []
                for block in item['children']:
                    if block['type'] not in ('paragraph', 'block_text'): raise ValueError('Flat paper reference lists only')
                    text.append(inlines(block['children']))
                body.append(r'\item '+' '.join(text))
            body.append('\\end{'+env+'}')
        else: raise ValueError('Unsupported block AST node: '+kind)
    if abstract: body.append(r'\end{abstract}')
    if caption is not None: raise ValueError('Unbound table caption')
    preamble = r'''\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage[review]{cvpr}
\usepackage{times,graphicx,amsmath,amssymb,booktabs,tabularx,array}
\usepackage{xurl}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=blue]{hyperref}
\def\confName{CVPR}\def\confYear{2027}\def\PaperID{DRAFT}
\title{'''+tex_escape(title)+r'''}\author{Anonymous}
\begin{document}\raggedbottom\maketitle
'''
    return preamble+'\n\n'.join(body)+'\n\\end{document}\n'


def actor_update():
    p = HOME/'moving_visual_actor_v1/analysis.json'
    if not p.exists():
        return ('The twelve-model comparison is registered and training is ongoing. '
            'A real 100+100 versus 200 recovery test and a full 4,000-update original-control '
            'refit are exact. No donor outcome comparison has been scored. '
            'No forecasting lift or component pass is claimed from those engineering checks.')
    a = read(p); t = read(p.with_name('training.json')); v = read(p.with_name('verification.json'))
    if t['models'] != 12 or v['models'] != 2 or not v['full_refits_exact']: raise ValueError('Report the complete actor matrix only')
    r = a['primary']; ci = r['interval95']
    return (f'All twelve actors complete 4,000 updates, with both fixed-design full refits exact. '
        f'The primary whole-twelve known difference is {r["known_pp"]:.3f} percentage points; '
        f'paired completion lower is {r["completion_lower_pp"]:.3f}, with exploratory source '
        f'interval [{ci[0]:.3f}, {ci[1]:.3f}]. '
        f'{r["positive_known_sources"]} of 14 source known contrasts are positive. '
        f'The registered component progression screen is {"supported" if r["component_progression_screen_pass"] else "not supported"}. '
        'This remains a one-seed, previously exposed A-side component experiment, not '
        'a complete policy risk result or independent confirmation. Deployment is unchanged.')


def actor_table():
    path = HOME/'moving_visual_actor_v1/analysis.json'
    if not path.exists():
        return ''
    a = read(path)
    lines = ['### Table 2. Source-balanced actor differences on exposed producer-A donors', '',
        '| Contrast | Prefix | Known difference (pp) | Completion lower (pp) |',
        '|---|---:|---:|---:|']
    names = {'primary': 'Motion - quality', 'quality_bundle': 'Quality - raw', 'motion_bundle': 'Motion - raw'}
    for contrast, name in names.items():
        rows = [r for r in a['sources'] if r['contrast'] == contrast]
        for i, prefix in enumerate((1, 3, 6, 12)):
            known = sum(r['summary'][i]['paired_known_pp'] for r in rows)/len(rows)
            lower = sum(r['summary'][i]['paired_completion_lower_pp'] for r in rows)/len(rows)
            lines.append(f'| {name} | {prefix} | {known:+.3f} | {lower:+.3f} |')
    lines += ['', 'The registered primary is motion minus quality over twelve steps.',
        'Raw comparisons are secondary bundle contrasts. The quality bundle has',
        'a small positive known difference, not a positive completion bound.',
        'Neither an early-prefix signal nor a secondary bundle replaces the',
        'failed primary. These are fourteen exposed development sources, not',
        'independent confirmation or three forecaster seeds.']
    return '\n'.join(lines)


def actor_abstract():
    path = HOME/'moving_visual_actor_v1/analysis.json'
    if not path.exists():
        return 'A separate image-conditioned actor comparison is reported with its actual execution status.'
    effect = read(path)['primary']['known_pp']
    return (f'Conditioning the trajectory actor itself also fails its matched motion '
        f'contrast ({effect:+.3f} percentage points on known labels). '
        'No new deployment or independent confirmation is established.')


def actor_mechanism():
    path = HOME/'moving_visual_actor_v1/branch_diagnostics.json'
    if not path.exists():
        return ''
    d = read(path)
    nonzero = all(r['fitting']['final_weight_norm'] > 0 for r in d['jobs'])
    worst = min(d['sources'], key=lambda key: d['sources'][key]['paired_known_pp'])
    q, m = (d['sources'][worst][arm] for arm in ('quality_geometry', 'motion_geometry'))
    return (f'Posthoc frozen-checkpoint diagnostics find all twelve final image '
        f'projections nonzero ({nonzero}). The largest known loss is source '
        f'{worst.rsplit("_", 1)[-1]}, where both arms have '
        f'{100*q["input_element_clipped_fraction"]:.3f}% and '
        f'{100*m["input_element_clipped_fraction"]:.3f}% clipped input elements, '
        f'respectively. Thus neither an entirely inactive branch nor input '
        'clipping alone explains that case. These diagnostics are not retrained '
        'ablations or proof of a single causal failure mechanism.')


def camera_update():
    folder = HOME/'moving_camera_frame_v1'
    path = folder/'analysis.json'
    if not path.exists():
        return ''
    training, verification = [read(folder/name) for name in ('training.json', 'verification.json')]
    replays = ROOT/'data/stage_cvpr2027_experiments/moving_camera_frame_v1/replays'
    if (training['models'] != 6 or verification['models'] != 1
            or not verification['full_refits_exact'] or not list(replays.glob('replay_*.json'))):
        raise ValueError('Camera evidence requires all six fits, one full refit and completed exact replay')
    a = read(path); p = a['primary']; lo, hi = p['interval95']
    lines = ['### 6.3 Explicit Query-View Geometry', '',
        'Six new actors complete the same 4,000 updates with no additional learned',
        'parameters. The full identity control and one fixed-design treatment',
        'refit match exactly; all twelve actor/camera-CV banks and reported',
        'scalars replay exactly. This is not six complete treatment refits.', '',
        f'The primary camera-actor minus identity known effect is {p["known_pp"]:+.3f}',
        f'percentage points. Its completion lower is {p["completion_lower_pp"]:+.3f}',
        f'with exploratory source interval [{lo:+.3f}, {hi:+.3f}].',
        f'{p["positive_known_sources"]} of fourteen known source contrasts are positive;',
        'the registered component screen is '+('supported.' if p['component_progression_screen_pass'] else 'not supported.'), '',
        '### Table 3. Query-view controls on exposed producer-A donors', '',
        '| Contrast | Prefix | Known difference (pp) | Completion lower (pp) |',
        '|---|---:|---:|---:|']
    for contrast, label in (('primary', 'Camera actor - identity'),
            ('camera_cv_effect', 'Camera CV - raw CV'), ('neural_over_camera_cv', 'Actor - camera CV')):
        rows = [r for r in a['sources'] if r['contrast'] == contrast]
        if len(rows) != 14:
            raise ValueError('Every camera contrast must retain fourteen source components')
        for i, prefix in enumerate((1, 3, 6, 12)):
            values = [[r['summary'][i][key] for r in rows] for key in ('paired_known_pp', 'paired_completion_lower_pp')]
            if any(v is None for group in values for v in group):
                raise ValueError('Undefined manuscript contrast must not be averaged over defined sources only')
            known, lower = [sum(group)/len(group) for group in values]
            lines.append(f'| {label} | {prefix} | {known:+.3f} | {lower:+.3f} |')
    lines += ['', 'The simple camera-CV control isolates whether geometric extrapolation',
        'alone is useful; its gain is not neural added value. All seven fixed',
        'causal baselines and all fourteen source components remain in the',
        'experiment report. No outcome-selected baseline replaces the CV floor.',
        'This remains a one-seed exposed-development component comparison, not',
        'an independent test or complete selected-policy risk certificate.',
        'No new controller or deployment follows from this table.']
    harm = [r['summary'][-1]['treatment']['known_easy_positive_harm_percent']
        for r in a['recordings'] if r['contrast'] == 'primary']
    fit_paths = sorted((folder/'fits').glob('*.json'))
    if len(harm) != 34 or len(fit_paths) != 6:
        raise ValueError('Complete recording and matched fitting diagnostics required')
    worse = 0
    for path in fit_paths:
        old = read(HOME/'moving_producer_crossfit_v1/fits'/path.name)['result']['losses'][-1]['loss']
        new = read(path)['result']['losses'][-1]['loss']
        worse += new > old
    lines += ['', f'The raw camera actor exceeds 2% known-easy positive harm in {sum(x is not None and x > 2 for x in harm)}',
        'of 34 recordings; this is a raw-actor diagnostic, not a full policy',
        f'certificate. In {worse} of six matched designs its final fitting-',
        'minibatch loss is also higher. That single-minibatch comparison is not',
        'a training-population average or proof of underfitting. Past alignment',
        'and future-affine restoration remain a bundled intervention; their',
        'separate effects are not identified by this experiment.']
    return '\n'.join(lines)


def figures(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.875, 2.0)); ax.set(xlim=(0, 10), ylim=(0, 3.1)); ax.axis('off')
    labels = ['Past trajectories\nand image proxy', 'Frozen causal\ncandidate forecasts', 'Learned benefit,\nharm and support', 'Joint query\nsubstitutions']
    xs = [.15, 2.7, 5.25, 7.8]
    for x, label in zip(xs, labels):
        ax.add_patch(Rectangle((x, 1.3), 2.0, 1.0, facecolor='#f4f6f8', edgecolor='#425466', linewidth=.8))
        ax.text(x+1, 1.8, label, ha='center', va='center')
    for x in xs[:-1]: ax.add_patch(FancyArrowPatch((x+2.03, 1.8), (x+2.52, 1.8), arrowstyle='-|>', mutation_scale=9, color='#425466'))
    ax.text(5.35, .3, 'Future labels: supervised costs / evaluation only, never inference inputs', ha='center', va='center', color='#995025', fontsize=7.5)
    ax.text(5.35, 2.75, 'Predicted budget feasibility is not realized-risk calibration', ha='center', va='center', color='#1f6f68', fontsize=7.5)
    fig.tight_layout(pad=.2)
    fig.savefig(BUILD/'decision_pipeline.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(BUILD/'decision_pipeline.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.25), layout='constrained')
    arms = ('no_image', 'image_bounds', 'quality_geometry', 'motion_geometry'); colors = ('#52616b', '#ba7628', '#745c9c', '#287c73')
    for i, (arm, color) in enumerate(zip(arms, colors)):
        p = a['policies'][arm]['joint_free']; u = p['utility']['known']; lo, hi = u['interval95']
        axes[0].errorbar(i, u['point'], yerr=[[u['point']-lo], [hi-u['point']]], fmt='o', color=color, capsize=3)
        axes[1].bar(i, p['safety']['supported_record_views'], color=color)
    for ax in axes:
        ax.set_xticks(range(4), ['None', 'Bounds', 'Quality', 'Motion'], rotation=15); ax.grid(axis='y', alpha=.2)
    axes[0].axhline(0, color='#555555', linewidth=.6); axes[0].set(ylabel='Known ADE gain (%)', title='Mean utility versus CV')
    axes[1].axhline(72, color='#b84c42', linestyle='--'); axes[1].set(ylabel='Supported views / 72', title='Original recording contract', ylim=(0, 76))
    fig.savefig(BUILD/'visual_cost_evidence.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(BUILD/'visual_cost_evidence.png', dpi=180); plt.close(fig)


def main():
    for name, expected in PINNED.items():
        if sha(HOME/name) != expected: raise ValueError('Changed frozen manuscript evidence: '+name)
    a = read(HOME/'moving_visual_risk_v1/analysis.json'); cal = read(HOME/'moving_visual_risk_v1/calibration_analysis.json')['summary']
    p = a['contrasts']['primary']['completion_lower_pp']; m = a['policies']['motion_geometry']['joint_free']; s = m['safety']
    primary_rows = a['contrast_rows']['primary']
    positive = sum(sum(r['completion_lower_pp'] for r in primary_rows if r['source'] == source) > 0 for source in set(r['source'] for r in primary_rows))
    table = ['| Inputs | ADE gain vs CV (%) | Hard gain (%) | Easy gain (%) | Original supported views | Worst signed easy loss (%) |', '|---|---:|---:|---:|---:|---:|']
    for arm, name in (('no_image', 'No image'), ('image_bounds', 'Bounds'), ('quality_geometry', 'Quality / geometry'), ('motion_geometry', 'Motion / geometry')):
        r = a['policies'][arm]['joint_free']
        worst = max(v['known_easy_net_degradation'] for x in a['rows'] if x['arm'] == arm
            for v in x['policies']['joint_free']['recordings'].values() if v['known_easy_net_degradation'] is not None)
        table.append(f'| {name} | {r["utility"]["known"]["point"]:.3f} | {r["hard_gain_percent"]:.3f} | {r["easy_gain_percent"]:.3f} | {r["safety"]["supported_record_views"]}/72 | {100*worst:.2f} |')
    values = dict(cost_table='\n'.join(table), actor_update=actor_update(), actor_table=actor_table(), actor_abstract=actor_abstract(), actor_mechanism=actor_mechanism(), camera_update=camera_update(), motion_lower_short=f'{p["point"]:.3f}',
        motion_lower_long=f'{p["point"]:.6f}', motion_ci_short=f'[{p["interval95"][0]:.3f}, {p["interval95"][1]:.3f}]',
        motion_ci_long=f'[{p["interval95"][0]:.6f}, {p["interval95"][1]:.6f}]', motion_positive_sources=positive,
        motion_known_short=f'{m["utility"]["known"]["point"]:.3f}', motion_pass=s['supported_record_views'],
        motion_signed_violations=s['easy_net_violations'], motion_easy_NA=s['undefined_easy_views'])
    for prefix, arm in (('quality', 'quality_geometry'), ('motion', 'motion_geometry')):
        values[prefix+'_easy_MSE'] = f'{cal[arm]["conditional_MSE"]["easy_harm"]:.6f}'
        values[prefix+'_Brier'] = f'{cal[arm]["observation_Brier"]:.6f}'
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False, keep_trailing_newline=True)
    markdown = env.from_string((PUBLIC/'manuscript.template.md').read_text()).render(**values)
    tex = latex_document(markdown); BUILD.mkdir(parents=True, exist_ok=True)
    (PUBLIC/'manuscript.md').write_text(markdown); (PUBLIC/'manuscript.tex').write_text(tex); (BUILD/'main.tex').write_text(tex)
    template = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v9/paper_build/cvpr.sty'
    shutil.copyfile(template, BUILD/'cvpr.sty'); figures(a)
    actor_paths = [HOME/'moving_visual_actor_v1'/name for name in
        ('registration_v2.json', 'training.json', 'verification.json', 'analysis.json',
         'execution_scope.md', 'resource_amendment.json')]
    actor_sources = {str(path.relative_to(ROOT)): sha(path) for path in actor_paths} if (HOME/'moving_visual_actor_v1/analysis.json').exists() else {}
    diagnostic = HOME/'moving_visual_actor_v1/branch_diagnostics.json'
    if diagnostic.exists():
        actor_sources[str(diagnostic.relative_to(ROOT))] = sha(diagnostic)
    camera_folder = HOME/'moving_camera_frame_v1'
    camera_paths = [camera_folder/name for name in ('registration_v2.json', 'training.json',
        'verification.json', 'control.json', 'analysis.json', 'forecast_freeze.json',
        'interruption_20261106T2054Z.json')]
    camera_paths += sorted((camera_folder/'fits').glob('*.json'))
    camera_paths += [HOME/'moving_producer_crossfit_v1/fits'/path.name
        for path in sorted((camera_folder/'fits').glob('*.json'))]
    camera_sources = {str(path.relative_to(ROOT)): sha(path) for path in camera_paths} if values['camera_update'] else {}
    write_json(PUBLIC/'source_bindings.json', dict(result_source='fresh_prose_typesetting_of_cached_verified_scalar_evidence',
        fixed_sources=PINNED, template_sha256=sha(PUBLIC/'manuscript.template.md'), builder_sha256=sha(Path(__file__)),
        markdown_sha256=sha(PUBLIC/'manuscript.md'), tex_sha256=sha(PUBLIC/'manuscript.tex'), cvpr_style_sha256=sha(template),
        actor_outcomes_included=(HOME/'moving_visual_actor_v1/analysis.json').exists(),
        actor_sources=actor_sources,
        camera_outcomes_included=bool(values['camera_update']), camera_sources=camera_sources,
        independent_confirmation=False, submission_ready=False, old_v9_sources_or_PDF_modified=False,
        stage5c_executed=False, smc_enabled=False))
    print('Native-stride working manuscript and private figures prepared; no new outcome scoring or submission claim.')


if __name__ == '__main__': main()
