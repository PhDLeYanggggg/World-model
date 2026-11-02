// Render the working revision via a Markdown AST; never modify the frozen v8.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
const require=createRequire(import.meta.url);
const { marked }=await import(pathToFileURL(require.resolve('marked')).href);
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const base=path.join(root,'outputs/publication_readiness_2026_09');
const output=path.join(base,'evidence_manuscript_v9');
const build=path.join(root,'data/stage_cvpr2027_experiments/evidence_manuscript_v9/paper_build');
const template=process.env.CVPR_TEMPLATE_DIR;
if(!template||!fs.existsSync(path.join(template,'cvpr.sty')))throw Error('Private official CVPR_TEMPLATE_DIR required');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
function writeIfChanged(file,value){
  const bytes=Buffer.isBuffer(value)?value:Buffer.from(value);
  if(!fs.existsSync(file)||!fs.readFileSync(file).equals(bytes))fs.writeFileSync(file,bytes);
}
const evidences={
  european_selection_controller_v1:['verification.json','all54_frozen_action_metric_groups_exact'],
  european_selection_cardinal_cost_v1:['verification.json','all54_metric_groups_exact'],
  european_selection_harm_deviance_v1:['verification.json','all54_metric_groups_exact'],
  european_calibration_source_fit_v1:['verification.json','all144_SOURCE_site_groups_exact'],
  european_independent_risk_calibration_v1:['verify.json','all47736_metric_groups_exact']
};
const bindings={};
const frozenV8=read(path.join(base,'evidence_manuscript_v8/build_sources.json'));
if(frozenV8.markdown_sha256!==sha(path.join(base,'evidence_manuscript_v8/manuscript.md'))
  ||frozenV8.tex_sha256!==sha(path.join(base,'evidence_manuscript_v8/manuscript.tex')))throw Error('Frozen v8 sources changed');
for(const [name,[file,key]] of Object.entries(evidences)){
  const home=path.join(base,name),p=path.join(home,'summary.json'),v=read(path.join(home,file));
  if(!v[key]||v.summary_sha256!==sha(p))throw Error('Complete verified evidence required: '+name);
  bindings[name]={summary_sha256:sha(p),verification_file:file,verification_sha256:sha(path.join(home,file))};
}
const figures=read(path.join(output,'figures/figure_sources.json'));
if(figures.evidence_summary_sha256!==bindings.european_selection_harm_deviance_v1.summary_sha256
  ||figures.public_svg_sha256!==sha(path.join(output,'figures/cost_loss.svg'))
  ||figures.private_PDF_sha256!==sha(path.join(build,'cost_loss.pdf')))throw Error('Current verified paper figure required');
const source=path.join(output,'manuscript.md'),tokens=marked.lexer(fs.readFileSync(source,'utf8'));
const title=tokens.find(t=>t.type==='heading'&&t.depth===1).text;
const symbols={'\\':'\\textbackslash{}','{':'\\{','}':'\\}','$':'\\$','&':'\\&','#':'\\#','%':'\\%','_':'\\_','^':'\\textasciicircum{}','~':'\\textasciitilde{}','<':'\\textless{}','>':'\\textgreater{}'};
const esc=s=>String(s).replace(/[\\{}$&#%_^~<>]/g,c=>symbols[c]);
function inline(items){return (items||[]).map(t=>{
  if(t.type==='text'||t.type==='escape')return t.tokens?inline(t.tokens):esc(t.text).replace(/\[([1-8])\]/g,'\\cite{ref$1}');
  if(t.type==='strong')return `\\textbf{${inline(t.tokens)}}`;
  if(t.type==='em')return `\\emph{${inline(t.tokens)}}`;
  if(t.type==='codespan')return `\\texttt{${esc(t.text)}}`;
  if(t.type==='link')return t.text===t.href?`\\url{${t.href}}`:`\\href{${esc(t.href)}}{${inline(t.tokens)}}`;
  if(t.type==='br')return '\\\\';
  throw Error('Unsupported inline node '+t.type);
}).join('');}
let abstract=false,references=false;
function blocks(items){let out='';for(const t of items){
  if(t.type==='space')continue;
  if(t.type==='heading'){
    if(t.depth===1)continue;
    if(abstract){out+='\\end{abstract}\n';abstract=false;}
    if(t.text==='Abstract'){out+='\\begin{abstract}\n';abstract=true;continue;}
    if(t.text==='LLM Usage Statement'){
      out+='\\clearpage\n\\typeout{M3W_MAIN_PAGES=\\number\\numexpr\\value{page}-1\\relax}\n\\section*{LLM Usage Statement}\n';continue;
    }
    if(t.text==='References'){references=true;continue;}
    const name=esc(t.text.replace(/^\d+(\.\d+)*\.?\s+/,''));
    out+=`\\${t.depth===2?'section':'subsection'}{${name}}\n`;continue;
  }
  if(t.type==='paragraph'&&t.tokens.length===1&&t.tokens[0].type==='image'){
    if(t.tokens[0].href!=='figures/cost_loss.svg')throw Error('Unbound paper figure');
    out+='\\begin{figure*}[!t]\\centering\\includegraphics[width=\\textwidth]{cost_loss.pdf}\n'+
      '\\caption{Matched harm-loss control on all six exposed development localities. Left: available-label ADE gain over the same floor. Right: direct paired completion-lower effect versus MSE control; dual is the registered primary. Intervals use 3,000 locality resamples after pipeline averaging. Improved net easy error does not repair failed positive-harm and support requirements.}\\end{figure*}\n';continue;
  }
  if(t.type==='paragraph'||t.type==='text'){out+=(t.tokens?inline(t.tokens):esc(t.text))+'\n\n';continue;}
  if(t.type==='code'){
    if(t.lang==='math')out+='\\begin{equation}\n'+t.text+'\n\\end{equation}\n';
    else out+='\\begin{verbatim}\n'+t.text+'\n\\end{verbatim}\n';
    continue;
  }
  if(t.type==='table'){
    const captions={Cohort:'Distinct data roles. Repeated forecasting-pipeline views do not increase the number of independent localities.',
      'Cost head':'Complete matched harm-loss readout. All arms retain the original recording roster and 2 percent contract. Undefined selected-easy support is not a safe zero.'};
    const caption=captions[t.header[0].text];if(!caption)throw Error('Unregistered table caption');
    const row=c=>c.map(x=>inline(x.tokens)).join(' & ')+' \\\\ \n';
    out+='\\begin{table*}[!t]\\centering\\small\n'+
      `\\begin{tabularx}{\\textwidth}{@{}${'X'.repeat(t.header.length)}@{}}\\toprule\n`+
      row(t.header)+'\\midrule\n'+t.rows.map(row).join('')+'\\bottomrule\\end{tabularx}\n'+
      `\\caption{${esc(caption)}}\\end{table*}\n`;continue;
  }
  if(t.type==='list'){
    if(references){out+='\\begin{thebibliography}{9}\n';for(const [i,item] of t.items.entries())out+=`\\bibitem{ref${i+1}} ${blocks(item.tokens)}`;out+='\\end{thebibliography}\n';}
    else{out+='\\begin{itemize}\n';for(const item of t.items)out+='\\item '+blocks(item.tokens);out+='\\end{itemize}\n';}
    continue;
  }
  throw Error('Unsupported block node '+t.type);
}return out;}
const tex=String.raw`\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage[review]{cvpr}
\usepackage{times,graphicx,amsmath,amssymb,booktabs,tabularx}
\usepackage{xurl}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=blue]{hyperref}
\def\confName{CVPR}\def\confYear{2027}\def\PaperID{DRAFT}
\title{`+esc(title)+String.raw`}\author{Anonymous}
\begin{document}\raggedbottom\maketitle
`+blocks(tokens)+'\n\\end{document}\n';
fs.mkdirSync(build,{recursive:true});writeIfChanged(path.join(output,'manuscript.tex'),tex);
writeIfChanged(path.join(build,'main.tex'),tex);writeIfChanged(path.join(build,'cvpr.sty'),fs.readFileSync(path.join(template,'cvpr.sty')));
const receipt={result_source:'fresh_working_prose_and_AST_typesetting_from_verified_scalar_evidence',
  builder_sha256:sha(fileURLToPath(import.meta.url)),
  source_bindings:bindings,markdown_sha256:sha(source),tex_sha256:sha(path.join(output,'manuscript.tex')),
  template_sha256:sha(path.join(build,'cvpr.sty')),figure_sources_sha256:sha(path.join(output,'figures/figure_sources.json')),
  frozen_v8_markdown_sha256:frozenV8.markdown_sha256,frozen_v8_tex_sha256:frozenV8.tex_sha256,
  frozen_v8_changed:false,current_forest_readout_included:false,compiled:false,independent_confirmation:false,submission_ready:false};
fs.writeFileSync(path.join(output,'build_sources.json'),JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt));
