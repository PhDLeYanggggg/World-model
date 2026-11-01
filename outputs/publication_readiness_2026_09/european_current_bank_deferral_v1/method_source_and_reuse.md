# Method Source And Reused Assets

Fresh source check:the official
[Mao,Mohri and Zhong ICML2024 paper](https://proceedings.mlr.press/v235/mao24d.html)
and local publisher PDF pages3-5were read,including Section4Eq3.
PDF SHA256:`06d3b0b82937a157b23084ba8ba6c7110420edec0ecd7e545f57fee18a9a1ed5`.
Its existing logistic-deferral implementation is reused,not rewritten as
one-hot best-expert classification. Eight fixed expert costs make the Eq3
coefficient for expert j equal to total cost minus its own cost. The baseline
has no extra inference-cost surcharge. The original bounded-loss/hypothesis
conditions are not verified for native partial ADE,query conditioning,alias
tying or the added frozen-risk filter;no consistency or recording-safety
guarantee is transferred. This is a disclosed application adapter,not a
reproduction of the original experiments or a new M3W method.

The prior old-bank experiment is cached_verified background only. The new
comparison regenerates causal scores from current bank/input identities and
trains on the identical full cyclic source query roster. The encoder is reused
from the same learned-reference warm states as the preceding own-choice study;
the new output heads start at zero. Both new loss arms have30,726active weights,
not the earlier critic's31,119. Three cost seeds still share forecaster43.

No current test outcome chooses an architecture,loss weight or guard threshold.
No independent selection/calibration/confirmation labels are opened. Third-party
PDFs remain private and excluded fromGit. Simulation,Stage5C and SMC are not run.
