# Amendment after the initial structural check

The initial identity-uniqueness check failed on Call Center before any
camera run or forecast scoring. The initial registration is retained.
If the same person ID has more than one row at a frame, discard all
rows at that ID/tick for support and scoring; do not guess which box
belongs to whom. Report the excluded rows and affected ID/ticks.
This is a reference-quality rule, not an online observation filter.
Clip grouping, the image-only pilot schedule and the 2% rule are unchanged.
