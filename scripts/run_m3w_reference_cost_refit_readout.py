"""Execute a minimal scope-name erratum without changing the frozen study files."""
import argparse
import ast
import copy
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_reference_cost_refit as run


def corrected_evaluation():
    source,line=inspect.getsourcelines(run.evaluate);tree=ast.parse(''.join(source));before=ast.dump(tree,include_attributes=False)
    calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='runtime']
    if len(calls)!=1 or 'runtime' not in run.evaluate.__code__.co_varnames:
        raise ValueError('Only the originally observed runtime-shadowing defect may be repaired')
    protected=calls[0];changed=[]
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='runtime' and node is not protected:
                changed.append((node.lineno,type(node.ctx).__name__));node.id='phase_receipt'
            return node
    tree=Rename().visit(tree)
    if len(changed)!=3 or sorted(v for _,v in changed)!=['Load','Load','Store']:
        raise ValueError('Exactly one receipt binding and two receipt uses must change')
    # Reversing the three renames must recover every original scientific AST node.
    restored=copy.deepcopy(tree)
    for node in ast.walk(restored):
        if isinstance(node,ast.Name) and node.id=='phase_receipt':node.id='runtime'
    if ast.dump(restored,include_attributes=False)!=before:raise ValueError('Non-name scientific change forbidden')
    after=ast.dump(tree,include_attributes=False);ast.increment_lineno(tree,line-1);ast.fix_missing_locations(tree)
    namespace=dict(run.__dict__);exec(compile(tree,str(run.FILES[0]),'exec'),namespace)
    fixed=namespace['evaluate']
    if 'runtime' in fixed.__code__.co_varnames or 'runtime' not in fixed.__code__.co_names:
        raise ValueError('Global runtime helper must no longer be shadowed')
    return fixed,dict(changed_name_occurrences=3,renamed_local_binding='runtime_to_phase_receipt',
        original_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        corrected_AST_sha256=hashlib.sha256(after.encode()).hexdigest(),
        original_AST_restored_exactly_after_inverse_rename=True,
        original_scientific_file_sha256=run.sha(run.FILES[0]),registration_sha256=run.sha(run.REG),
        executable_erratum_sha256=run.sha(Path(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('eval','verify'));p.add_argument('--resume',action='store_true');a=p.parse_args()
    fixed,binding=corrected_evaluation();receipt=json.loads((run.PUBLIC/'execution_erratum.json').read_text())
    if receipt['binding']!=binding:raise ValueError('Pre-readout erratum binding changed')
    run.parent.source.frozen.BASE.engine.committed(Path(__file__))
    run.parent.source.frozen.BASE.engine.committed(run.PUBLIC/'execution_erratum.json')
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fixed(a.phase=='verify',a.resume)


if __name__=='__main__':main()
