# Inference Checkpoint Directory Repair

Observed failure:infer PID94774 exits1 before its first allocation checkpoint
can be written. The original saver opens allocation/*.pt.tmp without creating
the parent directory. FileNotFoundError is an ordinary filesystem initialization
bug,not OpenMP/SHM,an interrupted training or a model-quality result.

No new model's policy readout occurred; no36-context choice freeze exists yet.
All96trained models and96complete exact refits are preserved. No model,loss,
feature,split,normalization,target,choice,budget,solver or risk-rule change.

The frozen training runner and registration remain byte-for-byte unchanged.
A small versioned safe entry point creates the allocation directory,then calls
the original runner and its original hash/commit checks. It checks its own
committed source and this note before execution. No checkpoint or raw input is
modified by directory initialization. Recovery remains original `--resume`.

Use the safe entry point for inference and replay. All other training/evaluation
commands can retain the original runner. This is a reproducibility repair,not
an outcome-driven model amendment or another independent experiment.
