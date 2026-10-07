# Registration transport repair

The first October 7 submission did not execute its remote Python code: the remote
shell returned `Argument list too long`. The registration remains byte-identical
and the submission-intent guard will independently reject any duplicate.

`submit_m3w_easy_harm_recovery_stream.py` sends the same argument list as one JSON
line on standard input, followed by the same small code archive. A short remote
stdlib preamble reconstructs `sys.argv` and executes the exact registered SUBMIT
string. No training, references, tolerances, scheduler parameters or acceptance
logic changes. No checkpoint is transferred to the local machine.
