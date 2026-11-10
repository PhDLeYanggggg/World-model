# Restore observed geometry before testing learned direction repair

The previous one-rotation rule used histories centered on each actor's
own last observed position. Its inverse-distance weights therefore
contained no actual between-actor spacing. Keep that run and all
original outcomes. Use the previously hashed geometry export to
repeat the rule with actual observed spacing and no other change.

Then compare two identical 29-input PyTorch dynamics networks. Both
learn the same bounded, heading-aligned 12-step displacement from
eight observed positions, under the same observed loss and missing
step penalty. One sees two nearest neighbors' past relative geometry
and velocity, while the other receives zeros for those inputs. Use
the same 200 updates, batch of 256, seed and source-balanced sampling.
Train only on the five donors for each held/calibration pair. No
threshold search: both use the previously fixed positive individual
and relative proxy gain rule. The matched control reuses precisely the
neighbor model's rotating actor and selects the angle from the
neighbor-free model's individual utility score.

Freeze all three seed runs and choices before held scoring. Original
actor choices, query, per-step displacement norm and 2% checks do not
change. Report direct conservative trajectory and relative motion
contrasts, pass counts, switches, model-only available ADE and
3,000 paired location draws. This is exposed OpenDD vehicle
development evidence, not a new external evaluation or release.
No Songdo, WEPDTOF or DroneCrowd outcomes are opened.
