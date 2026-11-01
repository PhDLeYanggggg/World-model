// Build an explicitly provisional anonymous draft using the standard Markdown AST.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
const require=createRequire(import.meta.url);
const { marked }=await import(pathToFileURL(require.resolve('marked')).href);
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const pub=path.join(root,'outputs/publication_readiness_2026_09');
const output=path.join(pub,'evidence_manuscript_v8');
const build=path.join(root,'data/stage_cvpr2027_experiments/european_own_decision_cost_v1/paper_build');
const template=process.env.CVPR_TEMPLATE_DIR;
if(!template||!fs.existsSync(path.join(template,'cvpr.sty')))throw Error('Provide the official private CVPR_TEMPLATE_DIR');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
function complete(name,flag){
  const dir=path.join(pub,name),c=read(path.join(dir,'complete.json')),v=read(path.join(dir,'verification.json'));
  if(c.summary_sha256!==sha(path.join(dir,'summary.json'))||v.summary_sha256!==c.summary_sha256||!v[flag])throw Error('Complete verified public result required');
  if(v.complete_sha256&&v.complete_sha256!==sha(path.join(dir,'complete.json')))throw Error('Readout identity changed');
  return read(path.join(dir,'summary.json'));
}
const projection=complete('european_coherent_cost_projection_v1','all72_projection_inference_and_frozen_action_scores_exact');
const ownReady=fs.existsSync(path.join(pub,'european_own_decision_cost_v1/verification.json'));
const own=ownReady?complete('european_own_decision_cost_v1','all72_new_cost_inference_and_frozen_action_scores_exact'):null;
const fmt=x=>x===null?'not estimable':Number(x).toFixed(2);
const ci=x=>x.point===null?'not estimable':`${fmt(x.point)} [${x.CI95.map(fmt).join(', ')}]`;
let amendment=`### Updated Coherent-Cost Controls\n\nThese later comparisons use the updated frozen neural bank and SOURCE fitting roster. They are not a ranking against the earlier bank's exposed inner-held results in Section 7. No independent outcomes are opened. All 149,510 targets, 74 recordings, 12 localities and 72 groups remain; three cost seeds share forecaster43.\n\nA zero-update control shares each agent's reference moments across causally available candidates and caps benefit at that reference. Harm and trajectories remain unchanged. This removes the cost contradictions but does not improve the full policy.\n\n| Projection contrast | Known utility, pp [95% CI] | Completion-lower, pp [95% CI] |\n|---|---:|---:|\n| Projected minus raw learned joint | ${ci(projection.effects.learned_projection.full.known_pp)} | ${ci(projection.effects.learned_projection.full.lower_pp)} |\n| Projected joint minus its independent policy | ${ci(projection.effects.projected_joint_minus_independent.full.known_pp)} | ${ci(projection.effects.projected_joint_minus_independent.full.lower_pp)} |\n\nOnly ${projection.safety.learned_joint.supported_record_views}/444 original recording-risk views pass after projection. The positive joint-allocation component does not rescue the failed primary or establish safety. Required empty post-hoc matched controls remain not estimable. These are nominal SOURCE fitting intervals, not independent confirmation.\n\nA further matched training experiment uses the same coherent decoder, warm weights, 31,119 active parameters and complete cyclic query draws in both arms. It changes only whether the aggregate signed-risk residual is supervised on fixed causal initial choices or detached current joint choices. All legal known pointwise cost labels remain supervised. Empty or partly unknown selected-query labels do not become supervised zero. The network's context input stays fixed in both arms. This is not differentiation through an integer solver or an implementation of a published consistency loss.\n\n`;
if(own){
  const r=own.safety.own_joint;
  amendment+=`| Own-decision contrast | Known utility, pp [95% CI] | Completion-lower, pp [95% CI] |\n|---|---:|---:|\n| Own joint minus fixed joint | ${ci(own.effects.own_minus_fixed.full.known_pp)} | ${ci(own.effects.own_minus_fixed.full.lower_pp)} |\n| Own joint minus frozen projection | ${ci(own.effects.own_minus_frozen.full.known_pp)} | ${ci(own.effects.own_minus_frozen.full.lower_pp)} |\n\nAll 36 heads finish 208,320 primary updates, plus 14,336 full-refit updates. Original controls and complete new-cost inference/frozen-action scores replay exactly. Own-choice risk support is ${r.supported_record_views}/444; its fitting advancement status is ${own.fitting_advancement}. This is not an independent or deployable result.\n\n`;
}else amendment+='The new trial is still training. Its complete policy evaluation and primary effect are not yet available; no improvement is claimed from partial losses or completed individual heads.\n\n';
amendment+='Paired completion bounds use disagreement between policies on unknown outcomes. Subtracting individual lower-bound scores is not the paired lower effect. Missing-label uncertainty is not observed harm.\n\n';
const basePath=path.join(pub,'evidence_manuscript_v7/manuscript.md');
const base=fs.readFileSync(basePath,'utf8');
if(base.includes('/Users/')||base.includes('PhDLeYang'))throw Error('Personal identity/path cannot enter anonymous prose');
let tokens=marked.lexer(base);
const status=tokens.findIndex(t=>t.type==='paragraph'&&t.text.startsWith('Anonymous research draft.'));
if(status<0)throw Error('Expected archived status paragraph');
tokens.splice(status,1,...marked.lexer('**Working draft. Exposed SOURCE evidence only; not independent confirmation or submission-ready.**\n\n'));
let at=tokens.findIndex(t=>t.type==='heading'&&t.text.startsWith('9. Reproducibility'));
if(at<0)throw Error('Expected frozen manuscript structure');
const additions=marked.lexer(amendment);
additions.push({type:'m3w_figure',raw:'\n![Complete SOURCE cost evidence](cost_evidence.pdf)\n\n'});
tokens.splice(at,0,...additions);
const title=tokens.find(t=>t.type==='heading'&&t.depth===1).text;
const map={'\\':'\\textbackslash{}','{':'\\{','}':'\\}','$':'\\$','&':'\\&','#':'\\#','%':'\\%','_':'\\_','^':'\\textasciicircum{}','~':'\\textasciitilde{}','<':'\\textless{}','>':'\\textgreater{}'};
const esc=s=>String(s).replace(/[\\{}$&#%_^~<>]/g,x=>map[x]);
function inline(items){return (items||[]).map(t=>{
  if(t.type==='text'||t.type==='escape')return t.tokens?inline(t.tokens):esc(t.text).replace(/\[([1-6])\]/g,'\\cite{ref$1}');
  if(t.type==='strong')return `\\textbf{${inline(t.tokens)}}`;
  if(t.type==='em')return `\\emph{${inline(t.tokens)}}`;
  if(t.type==='codespan')return `\\texttt{${esc(t.text)}}`;
  if(t.type==='link'){
    if(/[{}\\]/.test(t.href))throw Error('Unsupported URL characters');
    return t.text===t.href?`\\url{${t.href}}`:`\\href{${esc(t.href)}}{${inline(t.tokens)}}`;
  }
  if(t.type==='br')return '\\\\';
  throw Error('Unsupported inline node: '+t.type);
}).join('');}
let references=false,abstract=false;
function blocks(items){let out='';for(const t of items){
  if(t.type==='space')continue;
  if(t.type==='heading'){
    if(t.depth===1)continue;
    if(abstract){out+='\\end{abstract}\n';abstract=false;}
    if(t.text==='Abstract'){out+='\\begin{abstract}\n';abstract=true;continue;}
    if(t.text==='LLM Usage Statement'){out+='\\clearpage\n\\typeout{M3W_MAIN_PAGES=\\number\\numexpr\\value{page}-1\\relax}\n\\section*{LLM Usage Statement}\n';continue;}
    if(t.text==='References'){references=true;continue;}
    const text=t.text.replace(/^\d+(\.\d+)*\.?\s+/,'');
    out+=`\\${t.depth===2?'section':t.depth===3?'subsection':'paragraph'}{${esc(text)}}\n`;continue;
  }
  if(t.type==='paragraph'&&t.tokens.length===1&&t.tokens[0].type==='image'){
    if(t.tokens[0].href!=='../feature_cost_figures_20261029/neural_increment_and_record_risk.png')throw Error('Unregistered manuscript image');
    out+='\\begin{figure*}[t]\\centering\\includegraphics[width=\\textwidth]{historical_evidence.pdf}'+
      '\\caption{Historical fixed-bank controller evidence on exposed internal source records,not independent confirmation. New-bank SOURCE fitting comparisons below are a separate controlled study.}\\end{figure*}\n';continue;
  }
  if(t.type==='paragraph'&&t.text.startsWith('For candidate k define')){
    out+=String.raw`For candidate $k$, define baseline-relative benefit and positive harm by
\begin{align}
B_k&=[c(b,y;M)-c(p_k,y;M)]_+,\\
H_k&=[c(p_k,y;M)-c(b,y;M)]_+,\\
R&=c(b,y;M),\quad ER=E R,\quad EH_k=E H_k.
\end{align}
The easy event $E$ is a positive CV error below a threshold fixed from producer-fitting data.
Future labels define this supervised event for loss and evaluation only; they never identify easy examples for the deployed decision.
`+'\n\n';continue;
  }
  if(t.type==='paragraph'&&t.text.startsWith('For a selected set, the positive-harm ratios')){
    out+=String.raw`For selected agents $C$, the positive-harm ratios are
\begin{equation}
\frac{\sum_{i\in C}H_i}{\sum_{i\in C}R_i},\qquad
\frac{\sum_{i\in C}EH_i}{\sum_{i\in C}ER_i}.
\end{equation}
Net easy ADE change is different: gains on one easy row must not erase positive harm on another for this ratio.
The approved tolerance is $2\%$ on every required recording.
A zero denominator is not estimable, not a safe zero. Count-matched controls compare utility only, not safety.
`+'\n\n';continue;
  }
  if(t.type==='paragraph'&&t.text.startsWith('For any future point, the reverse triangle inequality')){
    out+=String.raw`For any future point, the reverse triangle inequality bounds error change by
\begin{equation}
\big|\|p_t-y_t\|-\|b_t-y_t\|\big|\leq\|p_t-b_t\|.
\end{equation}
Thus any nonempty masked ADE change is bounded by $D_k=\max_t\|p_{kt}-b_t\|$.
The implementation additionally retains the preceding CV disagreement envelope when larger.
`+'\n\n';continue;
  }
  if(t.type==='paragraph'||t.type==='text'){
    out+=(t.tokens?inline(t.tokens):esc(t.text))+'\n\n';
    if(t.type==='paragraph'&&t.text.startsWith('A further matched training experiment'))out+=String.raw`For risk channel $j\in\{\mathrm{all},\mathrm{easy}\}$ and selected set $C_q$, the query residual is
\begin{equation}
e_q^{j}=\frac{\sum_{i\in C_q}[(\widehat H_i^j-\delta\widehat R_i^j)-(H_i^j-\delta R_i^j)]}{\sum_{i\in C_q}s_i},
\end{equation}
where $\delta=0.02$, $s_i$ is the causal scale, and the easy channel uses $(EH,ER)$.
With fixed fitting RMS $\sigma$, the aggregate loss is
\begin{equation}
\mathcal L_q=\frac{1}{Q}\sum_{q=1}^{Q}\frac{V_q}{2}\|e_q/\sigma\|_2^2.
\end{equation}
Here $V_q$ is one only for a nonempty selection with all selected labels known; it is a loss mask, not an inference input.
The shared total is $\frac12(\mathcal L_H+\mathcal L_B+\mathcal L_q)+\mathcal L_R$,
retaining the pointwise harm, benefit and reference losses.
`+'\n\n';
    continue;
  }
  if(t.type==='list'){
    if(references){out+='\\begin{thebibliography}{9}\n';for(const [i,item] of t.items.entries())out+=`\\bibitem{ref${i+1}} ${blocks(item.tokens)}\n`;out+='\\end{thebibliography}\n';}
    else{const env=t.ordered?'enumerate':'itemize';out+=`\\begin{${env}}\n`;for(const item of t.items)out+='\\item '+blocks(item.tokens);out+=`\\end{${env}}\n`;}
    continue;
  }
  if(t.type==='table'){
    const n=t.header.length,row=c=>c.map(x=>inline(x.tokens)).join(' & ')+' \\\\ \n';
    const captions={
      'Fixed controller':'Historical frozen-bank exposed inner-held results. Known ADE gains are over the same floor; 186 recording/head views repeat 31 unique recordings. Different controller families are not a single-factor comparison.',
      'Additional controller':'Reference and static-context ablations on the same historical bank and exposed inner-held roster. No full policy supports every required recording.',
      'Projection contrast':'Updated-bank SOURCE fitting control with fixed forecasts and no new training. Paired 95 percent intervals use locality resampling; the joint component cannot replace the failed primary or certify risk.',
      'Own-decision contrast':'Complete updated-bank SOURCE training comparison. All planned arms and original risks are retained. These conditional development intervals are not independent confirmation.'
    };
    const caption=captions[t.header[0].text];if(!caption)throw Error('Missing scoped table caption');
    out+='\\begin{table*}[t]\n\\centering\\small\n'+`\\begin{tabularx}{\\textwidth}{@{}${'X'.repeat(n)}@{}}\\toprule\n`+
      row(t.header)+'\\midrule\n'+t.rows.map(row).join('')+'\\bottomrule\n\\end{tabularx}\n'+`\\caption{${esc(caption)}}\n`+'\\end{table*}\n';continue;
  }
  if(t.type==='m3w_figure'){out+='\\begin{figure*}[t]\\centering\n\\includegraphics[width=\\textwidth]{cost_evidence.pdf}\n'+
    '\\caption{Complete exposed SOURCE cost evidence. Left: known-label gain over the same floor. Right: the paired known and completion-lower contrast. '+
    (ownReady?'The current own-versus-fixed training comparison is shown.':'The completed zero-update reference projection is shown; the new training trial is not yet evaluated.')+
    ' Intervals are conditional SOURCE fitting evidence, not independent confirmation or recording-risk certification.}\\end{figure*}\n';continue;}
  if(t.type==='code'){out+='\\begin{verbatim}\n'+t.text+'\n\\end{verbatim}\n';continue;}
  if(t.type==='blockquote'){out+='\\begin{quote}\n'+blocks(t.tokens)+'\\end{quote}\n';continue;}
  if(t.type==='hr'){out+='\\medskip\\hrule\\medskip\n';continue;}
  throw Error('Unsupported block node: '+t.type);
}return out;}
const tex=String.raw`\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage[review]{cvpr}
\usepackage{times,graphicx,amsmath,amssymb,booktabs,tabularx}
\usepackage{xurl}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=blue]{hyperref}
\def\confName{CVPR}
\def\confYear{2027}
\def\PaperID{DRAFT}
\title{`+esc(title)+String.raw`}
\author{Anonymous}
\begin{document}
\raggedbottom
\maketitle
`+blocks(tokens)+'\n\\end{document}\n';
fs.mkdirSync(output,{recursive:true});fs.mkdirSync(build,{recursive:true});
fs.writeFileSync(path.join(output,'manuscript.md'),tokens.map(t=>t.raw||'').join(''));
fs.writeFileSync(path.join(output,'manuscript.tex'),tex);
fs.writeFileSync(path.join(build,'main.tex'),tex);fs.copyFileSync(path.join(template,'cvpr.sty'),path.join(build,'cvpr.sty'));
const receipt={base_manuscript_sha256:sha(basePath),projection_summary_sha256:sha(path.join(pub,'european_coherent_cost_projection_v1/summary.json')),
  own_result_status:ownReady?'cached_verified_complete_SOURCE':'not_run_complete_policy_while_training',
  own_summary_sha256:ownReady?sha(path.join(pub,'european_own_decision_cost_v1/summary.json')):null,
  markdown_sha256:sha(path.join(output,'manuscript.md')),tex_sha256:sha(path.join(output,'manuscript.tex')),
  official_template_sha256:sha(path.join(build,'cvpr.sty')),independent_confirmation:false,submission_ready:false,compiled:false};
fs.writeFileSync(path.join(output,'build_sources.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
