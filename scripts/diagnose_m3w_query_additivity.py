"""Reproduce the frozen fit-only gradient-additivity assertion failure."""
import json

from scripts import audit_m3w_query_conditioning as runner


def execute():
    reg, ancestor = runner.admission()
    _, models = runner.run.models(reg)
    api, audit = runner.run.api, runner.audit
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    context = sorted(reg['contexts'])[0]
    fit, held = runner.run.parent.run.parent.prior.packets(
        reg['contexts'][context], ancestor, lambda **kw: print(json.dumps(kw), flush=True))
    del held
    state = api.core.read_checkpoint(runner.run.ROOT/models[context, 43, 'self']['checkpoint']['path'])
    batch = audit.prepare_probes(fit, state)[6]
    result = {}
    for precision in ('float32', 'float64'):
        model = audit.model_from(state)
        if precision == 'float64':
            model.double()
            current = {s: {k: v.double() if isinstance(v, api.torch.Tensor)
                           and v.is_floating_point() else v for k, v in d.items()}
                       for s, d in batch.items()}
        else:
            current = batch
        parts, _ = audit.objective_parts(model, state, current)
        params = [p for p in model.parameters() if p.requires_grad]
        gradients = {k: audit.gradient_vector(v, params).double() for k, v in parts.items()}
        total = audit.gradient_vector(sum(parts.values()), params).double()
        error = (sum(gradients.values())-total).abs()
        magnitude = sum(v.abs() for v in gradients.values())
        old_bound = 2e-5+2e-4*total.abs()
        result[precision] = dict(
            parameters=len(total), old_mismatch_count=int((error > old_bound).sum()),
            max_abs_error=float(error.max()), error_norm=float(error.norm()),
            total_norm=float(total.norm()), component_magnitude_norm=float(magnitude.norm()),
            relative_component_norm_error=float(error.norm()/magnitude.norm()),
            max_component_scaled_error=float((error/(magnitude+1e-12)).max()),
            worst_coordinates=[dict(index=int(i), error=float(error[i]), total=float(total[i]),
                                    component_abs_sum=float(magnitude[i]),
                                    components={k: float(v[i]) for k, v in gradients.items()})
                               for i in api.torch.topk(error, 5).indices])
    doc = dict(result_source='fresh_run_targeted_fit_only_numerical_reproduction',
        context=context, seed=43, arm='self', batch_one_based=7,
        original_registration_sha256=runner.run.sha(runner.REG),
        script_sha256=runner.run.sha(__file__), result=result,
        model_changed=False, held_inference=False, independent_roles_read=False)
    # Path coercion is kept explicit for the shared immutable writer.
    runner.run.once(runner.PUBLIC/'additivity_reproduction.json', doc)
    print(json.dumps(doc, indent=2))


if __name__ == '__main__':
    execute()
