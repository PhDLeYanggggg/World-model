# OpenDD: does temporal stability supervision improve choice?

Use the exact donor actions, feature arrays and existing risk labels
from the prior exchange experiment. Add one conditional target:
among full-roster passes with three available background deletions,
did the option pass all three? Unknown support has no auxiliary loss.
Use the same 400-step optimizer, matched seed and initialization,
width and minibatch order in the control and auxiliary arms. The
control predicts the three full-recording risk and utility stages.
The new score multiplies full-pass probability by conditional
stability. Also rank the auxiliary model without the stability
multiplier to separate training from the decision score.

Freeze the aligned model's 101 intervention views. Pick once among
the same candidates, keeping 780 edits over 606 views, the original
2% check, motion caps and the prediction window. Primary: original
dual-pass rate versus the equal-architecture control and exact
same-view chance; secondary: stable rescues. Report paired location
bootstrap intervals with 3,000 draws after averaging seeds.
This is a nested development comparison, not external calibration or
confirmation. No new external outcomes or release threshold changes.
