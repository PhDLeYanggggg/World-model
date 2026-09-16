# Fixed-Head Execution Optimization

Before any v3 development evaluation, the real fit-only pilot completed 100 CPU
updates in 111.55 s, then resumed the same checkpoint for 100 MPS updates in
30.19 s (141.75 s cumulative). All losses were finite but highly variable;
the first CPU batch loss was 1.344 billion and step 200 loss was 12.097. Different
batches are not a convergence curve or evidence of prediction improvement.
The complete full/fold three-seed EqMotion budget alone extrapolates to about
10.1 hours on MPS, before cost-head fitting, inference and the matched local arm.

The pinned author core still evaluates all 20 output heads although only head 0
is used for the loss. v4 skips the 19 unused forward branches and changes only
the corresponding result reshape dimension. The selected head's arithmetic,
parameters, initialization, training budget and loss do not change. The original
third-party files remain untouched; the explicit AST adaptation and dependency
hash are recorded. Tests compare selected-head outputs and every trainable
gradient against the unpruned core, including nonzero fixed-head indices.

This is compute optimization, not a smaller model or a shortened experiment.
The v3 pilot stays at 200/10,000 updates and is explicitly incomplete; it is not
a model result. v4 begins a fresh fit from registered seeds instead of rewriting
old checkpoint identities. The matched Transformer is unchanged from its v3
configuration. Both v4 arms retain all scientific choices in the v3 decision,
including historical development exposure and no independent confirmation.

An additional fit-only 100-step MPS cost pilot is part of v4's registered budget
and continues by resume. No development result is used to select this execution
optimization. Source v3 is retained at commit 58cb19d2. Source/weight replay must
use matching identities, never edited historical bindings. Stage5C/SMC stay off.
