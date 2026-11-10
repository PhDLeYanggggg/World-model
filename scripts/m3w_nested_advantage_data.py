"""Nested donor data with explicit roots; never rebind frozen runner globals."""

from collections import Counter
from pathlib import Path

from scripts import run_m3w_opendd_onpolicy_donors as prior

run, study, exp = prior.run, prior.study, prior.exp
parents, old, roles, np, torch = prior.parents, prior.old, prior.roles, prior.np, prior.torch


class NestedData:
    def __init__(self, root, registration, context, beat):
        self.root, self.reg, self.context, self.beat = Path(root), registration, context, beat
        self.sites, self.banks = prior.load_data()
        self.heads, self.assets = context[6], context[10]
        self.prior_reg = run.sha(prior.OUT/'registration.json')
        self.updates, self.maps = 0, {}

    def rank(self, omitted, seed):
        omitted = sorted(omitted)
        roles.fit_roster(self.sites, omitted)
        name = 'rank_'+'_'.join(omitted)+f'_{seed}'
        builder = lambda: study.prior.model.SwapValue(False)
        if len(omitted) == 2:
            meta = self.heads['rank_heads'][name]
            expected = self.heads['registration']
        elif len(omitted) == 3:
            meta = run.read(prior.PRIVATE/'producers'/f'{name}.json')
            expected = self.prior_reg
        elif len(omitted) == 4:
            expected = self.reg
            path = self.root/'producers'/name
            identity = dict(registration=self.reg, omitted=omitted,
                fit=roles.fit_roster(self.sites, omitted), seed=seed, kind='nominee_ranker')
            if path.with_suffix('.json').exists():
                meta = run.read(path.with_suffix('.json'))
                if meta['identity'] != identity:
                    raise ValueError('Rank identity changed')
            else:
                cut = prior.cuts(self.sites, identity['fit'])
                table = [study.prior.prepare(self.sites[s], self.banks[s], cut[0]) for s in identity['fit']]
                net, meta, n = study.fit(identity, str(path), seed, builder, study.RANK_STEPS,
                    lambda rng: study.prior.batch(table, rng),
                    lambda m, x, y, pair, size: study.prior.model.objective(m(x, pair, size), y, True),
                    False, self.beat)
                self.updates += n
                run.once(path.with_suffix('.json'), run.stable(meta))
                return net, meta
        else:
            raise ValueError('Unregistered rank exclusion level')
        roles.verify_role(meta['identity'], self.sites, omitted)
        if meta['identity']['registration'] != expected or meta['identity']['seed'] != seed:
            raise ValueError('Rank seed or registration changed')
        return study.load(meta, builder, study.RANK_STEPS), meta

    def count(self, omitted, seed):
        omitted = sorted(omitted)
        fit = roles.fit_roster(self.sites, omitted)
        name = 'count_'+'_'.join(omitted)+f'_{seed}'
        if len(omitted) == 2:
            receipt = prior.PRIVATE/'producers'/f'{name}.json'
            expected = self.prior_reg
        elif len(omitted) == 3:
            receipt = self.root/'producers'/f'{name}.json'
            expected = self.reg
        else:
            raise ValueError('Unregistered count exclusion level')
        role = dict(registration=expected, omitted=omitted, fit=fit, seed=seed, kind='recording_count')
        if receipt.exists():
            meta = run.read(receipt)
            if meta['role'] != role:
                raise ValueError('Count producer role changed')
            roles.verify_role(meta['model']['identity'], self.sites, omitted)
            for dep in meta['dependencies'].values():
                run.checked(dep)
            return study.load(meta['model'], lambda: study.model.CountHead(meta['width']), study.HEAD_STEPS), meta
        if len(omitted) != 3:
            raise ValueError('Missing frozen pair producer')
        table, dependencies, audits = [], {}, {}
        for target in fit:
            excluded = sorted((*omitted, target))
            rank, rm = self.rank(excluded, seed)
            inner_fit = roles.fit_roster(self.sites, excluded)
            identity = dict(registration=self.reg, omitted=omitted, label_site=target,
                rank=rm['artifact'], cutoff_fit=inner_fit, seed=seed)
            path = self.root/'teachers'/('_'.join(omitted)+f'_{seed}_{target}.npz')
            def build():
                menu = study.menus(self.sites[target], self.banks[target], rank)
                labels, tally = study.labels(self.sites[target], self.banks[target], menu,
                                            prior.cuts(self.sites, inner_fit))
                return dict(**menu, local=labels[0], recording=labels[1]), tally
            arrays, meta, dependencies[target] = prior.cache(path, identity, build)
            table.append(({k: arrays[k] for k in ('x', 'legal', 'row', 'arm')},
                          (arrays['local'], arrays['recording'])))
            audits[target] = meta['audit']
            self.beat(phase='teacher', omitted=omitted, target=target, seed=seed)
        width = table[0][0]['x'].shape[-1]
        net, meta, n = study.fit(dict(**role, teachers=dependencies), str(receipt.with_suffix('')),
            seed, lambda: study.model.CountHead(width), study.HEAD_STEPS, study.batches(table, 1),
            lambda m, x, legal, y: study.model.objective(m(x, legal), y), False, self.beat)
        self.updates += n
        saved = dict(role=role, width=width, dependencies=dependencies, teacher_audit=audits, model=meta)
        run.once(receipt, run.stable(saved))
        return net, saved

    def packet(self, excluded, donor, seed):
        excluded = sorted(excluded)
        if donor in excluded or len(excluded) not in (1, 2):
            raise ValueError('Invalid nested donor role')
        omitted = sorted((*excluded, donor))
        before = self.updates
        count, cm = self.count(omitted, seed)
        rank, rm = self.rank(omitted, seed)
        if self.updates != before:
            raise ValueError('Producer training must finish before selector fitting')
        d, bank = self.sites[donor], self.banks[donor]
        plain = parents.prefix.menu_for(d, bank, rank)
        width = plain[0].shape[-1]
        if width != cm['width']:
            raise ValueError('Recipient and count feature schemas differ')
        if donor not in self.maps:
            self.maps[donor] = exp.road.raster_for(donor, self.assets[donor])[0]
        menu, _ = exp.augment(d, bank, plain, self.maps[donor])
        fit = roles.fit_roster(self.sites, omitted)
        cut = prior.cuts(self.sites, fit)
        identity = dict(registration=self.reg, excluded=excluded, donor=donor, seed=seed,
            count=cm['model']['artifact'], rank=rm['artifact'], cutoff_fit=fit, label='full_grid')
        file = self.root/'packets'/('_'.join(excluded)+f'_{donor}_{seed}.npz')
        def build():
            x, legal, rows, path, _ = menu
            with torch.inference_mode():
                logits = count(torch.from_numpy(plain[0]), torch.from_numpy(legal)).numpy()
            move = parents.base.model.movement_prefix(bank[0], path, rows)
            rng = np.random.default_rng(seed+913)
            feats, original, full, anchors, lengths, records = [], [], [], [], [], []
            tally = Counter()
            for rec in sorted(set(d['rec'])):
                ai, qi, r = old.indices(d, rec, rows)
                anchor, cap, status = roles.score_anchor(logits[qi], legal[qi], move[qi],
                                                       d['recordings'][qi], d['starts'][qi])
                tally.update(recordings=1, **{status: 1})
                if anchor is None:
                    continue
                pairs = old.model.exchange_menu(anchor, legal[qi], move[qi], cap)
                if not len(pairs):
                    tally.update(no_exchange=1)
                    continue
                if len(pairs) > old.SAMPLE:
                    pairs = pairs[np.sort(rng.choice(len(pairs), size=old.SAMPLE, replace=False))]
                feat = old.model.features(x[qi], legal[qi], move[qi], anchor, cap, pairs)[0]
                # Only supervision below this boundary reads outcomes; the legal slate is already fixed.
                cv, y, valid = bank[0][ai], d['y'][ai], d['valid'][ai]
                if np.linalg.norm(cv-y, axis=-1)[valid].sum() <= 0.:
                    tally.update(zero_reference_mass=1)
                    continue
                def label(plan):
                    p, s = old.measure(d, ai, bank, path, r, plan, cut)
                    return bool(s['dual_positive']), bool(prior.previous.prior.prior.certificate(
                        cv, p, y, valid, cut[0], s)[0])
                anchors.append(label(anchor))
                targets = [label(old.model.exchanged(anchor, *p)) for p in pairs]
                original.extend(v[0] for v in targets)
                full.extend(v[1] for v in targets)
                feats.append(feat)
                lengths.append(len(pairs))
                records.append(str(rec))
            f = np.concatenate(feats) if feats else np.empty((0, 4*x.shape[1]+9), np.float32)
            px, py, valid = prior.previous.prior.padded(torch.from_numpy(f),
                torch.tensor(original, dtype=torch.bool), torch.tensor(full, dtype=torch.bool), lengths)
            px = exp.design(px.flatten(0, 1), width, 'aligned').reshape_as(px)
            ax = prior.model.with_incumbent(px)
            a = np.asarray(anchors, bool).reshape(-1, 2)
            mask = np.column_stack((np.ones(len(a), bool), valid.numpy()))
            labels = np.concatenate((a[:, None], py.numpy()), axis=1)
            tally.update(slates=len(a), options=int(mask.sum()), original_positive=int(labels[..., 0][mask].sum()),
                         full_positive=int(labels[..., 1][mask].sum()))
            if not np.isfinite(ax.numpy()).all() or (labels[..., 1] & ~labels[..., 0]).any():
                raise ValueError('Invalid causal features or full-grid labels')
            return dict(x=ax.numpy(), valid=mask, original=labels[..., 0], full=labels[..., 1],
                        recordings=np.asarray(records, dtype='U128')), dict(tally)
        arrays, meta, receipt = prior.cache(file, identity, build)
        return tuple(torch.from_numpy(arrays[k]) for k in ('x', 'full', 'valid')), receipt, meta['audit']
