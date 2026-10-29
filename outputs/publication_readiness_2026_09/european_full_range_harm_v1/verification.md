# Execution and Verification

Result sources: fresh18 training continuations and144 new inference views;
cached_verified six comparator models and original TRAIN input packets;
independent confirmation not_run, because the method has not passed development.

PilotPID47366 exit0:200 updates,13.781193s,2,200,584,192-byte peak RSS; exact
resume and protected outputs. FullPID47588 exit0:18/18,36,000 updates,
185.181761s. All finals frozen before readoutPID48100, exit0,144/144 views,
112.766226s. No training/evaluation process remains running.

Registration SHA256:
`413a26501a4a616907af38d07a2a5efc9c6468dc87aaf6059f307e3a445b05ce`

Training freeze SHA256:
`823ec3757d01c8218349022569632917718cfd4b5494767acaed5938cdddf92b`

Readout summary SHA256:
`49e3109b4e0c56fb076f6a4e524bfbd80dabfc743e96852a5fb7ad4c84f9b431`

Checks verify complete identities, checkpoint hashes, parent control prediction
hashes, same initialization/sampler draws/steps/parameter count, protected base,
all18 initial real objective values, selected counts, missing-label handling,
completion arithmetic, paired matched counts and regenerated report text.
97 scoped tests passed in5.05s. Tests include numerical bounds, near-zero anchor
escape, finite zero-envelope
backward, exact resume, unknown source rejection, corrupted arithmetic rejection
and honest negative/undefined reporting. Scoped testing is used; full legacy
pytest is not run because some old tests overwrite frozen scientific artifacts.

Figure was rendered and visually inspected: no cropped labels or omitted
negative intervals. It remains under the ignored private experiment directory;
the public figure receipt records its hash. No raw trajectories, caches, images,
checkpoints or third-party data are submitted to Git.

This is same-agent verification, not an independent blinded review. Engineering
checks passing does not override the failed research gate. No new scheduler job,
authentication change, independent-role access, Stage5C or SMC.
