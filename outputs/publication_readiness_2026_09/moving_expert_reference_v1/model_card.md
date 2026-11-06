# Shared Frozen-Expert Reference Model Card

**Status: research-only; no deployment change.** Neither matched arm passes the
original recording risk/support requirements. The registered utility interval
crosses zero. This is a neural reference-cost model, not new trajectory dynamics,
JEPA pretraining or a complete world-model training result.

## Inputs And Architecture

Each reference model has3,938parameters: inherited history50->32 plus additive
context70->32 without bias, SiLU and32->2 output logits. The new context weights
start at zero; both arms inherit the same reference weights. The control masks
the extra input, the treatment uses descriptors of the fixed full-amplitude
expert rollout. Both have the same parameterization, though masking naturally
changes the information and active feature weights.

All features are past-derived or predicted by a frozen causal expert. A predicted
rollout endpoint is not a future target endpoint. Context is joined exactly by
the original agent-window/action IDs, predicted once per current agent row and
broadcast to its six legal actions. Table "agents" counts mean agent-window rows,
not unique tracks. The first50 history/neighbor descriptors match every action.

Conditional R is softplus; conditional ER is R times a sigmoid gate. Frozen
benefit is capped at new R, while conditional/composed harm and easy harm plus
released-label probability stay exact per candidate. Benefit can incur finite-
precision recomposition differences in addition to the necessary cap. Outputs
are expected contract contributions, not calibrated conservative bounds.

## Training

Two arms x eight source folds x three risk seeds17/29/43 =48new models. Each
uses4,096AdamW updates; all48complete fits repeat exactly. Parent weights and
reference normalization/RMS are frozen, context normalization is learned only
from known fitting rows, and the query's entire source is excluded. Training
loss is observed-only R/ER normalized MSE. Unknown labels are not zero targets.
The two arms share sampling/initialization, final checkpoints are fixed, and no
threshold is selected from evaluation outcomes. Teacher seed remains17.

The source-held cost collection is not one universal deployment weight. It
requires its corresponding frozen parent, normalization, context schema and
causal action bank. The private checkpoints are not shipped in Git. Follow
operation_zh.md; code download alone does not reproduce authorized private data.

## Results

Expert context known ADE gain vs CV is0.790637%, but the registered paired
completion-lower effect vs history-refit is+0.133479pp with exploratory source
interval[-0.140981,+0.556238]. Only24/72 required repeated recording views pass;
26have known-easy positive-harm violations. Total reference MSE improves while
easy-reference MSE worsens. The model is not a safe deployment candidate.
Worst defined signed easy-net degradation is9.0168%, and worst defined gross
easy risk is40.2986%; six easy ratios remain undefined. Tail reductions versus
frozen control do not substitute for the unchanged complete2%requirements.

## Limitations And Data Roles

Evaluation is already-exposed own-B with62,466past-eligible rows in8sources and
24recordings, repeated over3risk seeds, not72independent scenes. Training donors
add125,540agent-window rows. No new independent C/H is used. Future masks/labels
are loss/evaluation only. No central velocity, future endpoint/goal input, test
endpoint goals or test normalization. Parent lineage is verified by the existing
source contracts; upstream annotation-generation causality remains unverified.

The allocator optimizes predicted risk/utility across eligible contemporaneous
rows, not pairwise physical collisions or necessarily all visible scene agents.
Available-label obs8/pred12 image-coordinate ADE is not complete12ADE, metric
distance, seconds, historical SDD t50, true3D or foundation evidence. Labels are
not human gold. Original24-source-group main remains NA. Stage5C/SMC stay off.
