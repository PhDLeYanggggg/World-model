# Bounded Native-Node Accounting Comparison

Both jobs37985561 and37985676 landed on the same virtual compute node.
System Python3.10.12 and module Python3.11.6 both failed to access historical
Slurm accounting. Removing the explicit cluster argument did not repair sacct;
local squeue succeeded with no active rows for the three fixed owned jobs.
This is evidence against a module-only explanation, not proof of a global
database outage or successful historical terminal status.

One further isolated comparison uses the already-used zen3 compute-node class,
with the same one-CPU, one-GiB, five-minute bounds and two interpreter branches.
The worker refuses a virtual node under this variant. It uses a new exclusive
output namespace, retaining both earlier observations unchanged. The source
change is limited to namespace and node guards; the query commands are unchanged.
No shared environment, system service, credentials, network configuration,
training, numerical inference or original job is changed.

If this comparison also fails, do not submit unchanged query jobs again.
Preserve the concrete scheduler-access blocker and keep scientific readout
not_run pending an admissible transport remedy. No successful join row is
fabricated, and no training is duplicated. Stage5C and SMC stay off.
