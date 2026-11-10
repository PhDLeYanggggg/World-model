# Supplement: exact one-edit exchange and matched chance

Extract into an empty directory and run `python3 -I -B -S verify.py`.
The script checks the earlier manuscript evidence and recalculates the
exact exchange and chance-control summaries. It needs no network, original
tracks, model weights or Python packages.

The original 606 development views use the same 780 edits, movement caps,
chronological credits, candidate paths and 2% recording rule. Exhaustively
scoring all 7,198 legal single exchanges (with the unchanged anchor also
allowed) gives 181 positive views. Exactly 68 exceed 0.10 percentage points
against both references, four exceed 0.50, and none exceeds 1.00.

The guarded history-feature model gives 55 dual passes, versus 60 for no
exchange. It rescues 10 and destroys 15. Forced selection gives 60 dual
passes, with 30 rescued and 30 destroyed. The matched chance controls act
on exactly the same views as their respective model arms and average
uniformly over the legal exchanges, without a Monte Carlo approximation.
Neither model has a clear advantage over this control in the paired
location intervals. This is offline development evidence, not a streaming
policy or an independent confirmation result.
