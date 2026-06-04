# Stage44 Failure Analysis

- best variant: `hybrid_no_scene`
- no-baseline collapse: `False`
- JEPA downstream lift: `False`
- scene lift: `False`
- interaction lift: `False`
- Transformer/SSM lift: `True`
- t100 still negative: `True`
- repair actions executed: `False`

## Next Repair Points

- replace JEPA target with multi-component future world-state latent and stronger masked trajectory/interaction targets
- audit scene proxy sparsity and require nonzero scene-token intervention in validation
- upgrade static graph features to dynamic graph tokens with interaction-risk auxiliary supervision
- train horizon-specific latent dynamics with K=64/128 history before making any t100 claim
