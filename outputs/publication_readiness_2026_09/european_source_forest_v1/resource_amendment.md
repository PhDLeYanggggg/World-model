# Resource-Only Preflight Amendment

The first16-tree pilot completed on actual optimization rows in4.91 fitting
seconds,4.84GB peak RSS; compressed checkpoint1,472,139 bytes. No validation or
transfer outcome was read in that pilot. The original176-byte/node conservative
storage estimate exceeded the available space above10GiB by about54MB, so the
original training entry point correctly stopped before starting any full fit.
The failure and original pilot report are preserved.

Before further fitting, inspect actual sklearn tree array layouts. Replace only
the storage estimate's guessed176-byte/node payload by its measured nodes/value
array dtype sizes. Keep worst-case node count for every source,128trees, two
largest additional copies,1% codec slack,1MiB per fit metadata and32MiB reporting.
This estimate does not extrapolate the observed compression ratio. If still too
large, stop for CREATE. Keep the original10GiB runtime guard. No cleanup of prior
data, smaller training, target change, model selection or threshold adjustment.

The separately bound retry wrapper calls the unchanged registered fit function
and resumes the original pilot checkpoint. Original scientific registration,
source partitions, validation rule and model hashes remain valid. The wrapper
and this resource amendment must be committed before the full run.
