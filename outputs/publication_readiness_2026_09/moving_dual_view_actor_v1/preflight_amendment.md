# Before-Execution Signal Snapshot Reference Fix

The initial draft,code and seven passing focused checks are preserved in
Git21c3057e. No real admission,pilot,control or treatment ran with that draft.
The model,trainer,tests,config,protocol and numeric evaluation are unchanged.

The real signal pilot records a committed step1checkpoint before resuming the
same trajectory to200. Its proof must point to an immutable copy,not the rolling
checkpoint that the resume overwrites. The runner now copies the step1snapshot
before recording its artifact hash and revalidates it before comparing200states.
No extra optimizer updates,data access,model parameters,loss,sampler,budget,
primary/risk/support/NA rules or result selection are introduced.

Active registration isregistration_v2.json and input_admission_v2.json. The
original registration.json/input_admission.json remain unchanged;all scientific
implementation files must match their original draft hashes. This is a proof-
reference correction before real fitting,not a post-outcome model adjustment.
