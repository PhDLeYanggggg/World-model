# Prepare the complete first day without choosing evaluation roles

Use the publisher's complete 2022-10-04 intersection inventory,
including all ten flight CSVs per archive. The rule is the same
acquisition day as the original four-site audit, not a selection by
forecast error. Check publisher size and MD5; store a local SHA-256
for every archive. Reuse the original A/B/C/E metadata and observed
pack after hash verification. Process the 16 new sites under the
same clock, duplicate-frame and visibility rules.

Select nonoverlapping windows with eight observed and twelve
prediction steps on the existing native grid. Use only coordinate
values from the selected past rows to build features. The future
availability and source row pointers stay in a separate label index.
No future coordinates are parsed. Report pair and complete-roster
coverage separately; a pair with labels does not give labels to
the whole roster.

Before assigning roles, merge sites linked by published map
cutout overlap, observed footprint overlap, overlapping same-drone
clock spans, apparent duplicate members, or matching observed
prefixes. Screen near-synchronous observed prefixes only within
overlapping observed envelopes. These are candidate source
components, not proof of independence for a vehicle that travels
between nonoverlapping sites. Do not use the number of windows,
flight files or dates as the independent sample count.

Build unscored CV, OLS4, median4 and vehicle-completion predictions
on the exact same observed inputs. Average the seven location-
excluded OpenDD vehicle models separately within each of the
three existing seeds. Use the completion-trained candidate
already chosen on OpenDD. The vehicle bank does not open the
label index or choose a policy; equal-radius counts measure
available actions only, not gains or passing risk checks.

No calibration or test assignment, forecast scoring, deployment,
reserved DroneCrowd evaluation, Stage5C or SMC. The eventual
Songdo role assignment is a separate prospective decision and
retains the original 2% policy checks.
