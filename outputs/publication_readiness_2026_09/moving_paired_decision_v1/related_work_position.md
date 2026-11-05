# Method Positioning

Pairwise neural logistic ranking is established, not a new invention.
Burges et al., Learning to Rank using Gradient Descent, ICML2005, defines a
logistic cross-entropy on differences of model scores. I checked the primary
method equations1-3; this is a method-section check, not full proof verification.
[Primary paper](https://icml.cc/Conferences/2005/proceedings/papers/012_LearningToRank_BurgesEtAl.pdf).

This controlled test applies a temperature-scaled pairwise term to the maximum
of existing intervention risk/decision outputs, using observed supported
harmful/beneficial pairs from the same history query. A matched pointwise arm
has the same value and score gradient at zero, the same pair weights and the
same base loss. The experiment asks whether relative discrimination improves
fixed-rule decisions; it does not claim a novel ranking loss, calibrated safety
probabilities, or new trajectory dynamics. Ranking alone cannot determine the
zero acceptance boundary, so the original supervised anchors are retained.

If successful, this would be component evidence for baseline-relative
intervention learning, not by itself a top-conference contribution. Independent
scene confirmation, three producer seeds, all original safety/support gates,
matched intervention controls and the full method comparisons remain necessary.
Negative results must remain in the evidence package.
