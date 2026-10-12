# Bounded Native-Node Accounting Comparison

Both jobs37985561 and37985676 reported the same operating-system hostname.
System Python3.10.12 and module Python3.11.6 both failed to access historical
Slurm accounting. Removing the explicit cluster argument did not repair sacct;
local squeue succeeded with no active rows for the three fixed owned jobs.
This is evidence against a module-only explanation, not proof of a global
database outage or successful historical terminal status.

One further isolated comparison uses the already-used zen3 compute-node class,
with the same one-CPU, one-GiB, five-minute bounds and two interpreter branches.
The requested constraint does not by itself establish physical machine type.
The corrected worker records Slurm nodelist separately from OS hostname. It uses a new exclusive
output namespace, retaining both earlier observations unchanged. The source
change is limited to namespace and node guards; the query commands are unchanged.
Job37985944 exited before querying because the first implementation wrongly
required the OS hostname to begin erc-hpc-comp. This was a worker guard error,
not evidence against the database hypothesis. The repaired guard still requires
a CREATE compute allocation, but no longer equates OS hostname and Slurm node
name. The failed run never created the exclusive output directory; a separately
identified corrected execution is allowed, not a retry of unchanged code.

No shared environment, system service, credentials, network configuration,
training, numerical inference or original job is changed.

If this comparison also fails, do not submit unchanged query jobs again.
Preserve the concrete scheduler-access blocker and keep scientific readout
not_run pending an admissible transport remedy. No successful join row is
fabricated, and no training is duplicated. Stage5C and SMC stay off.
