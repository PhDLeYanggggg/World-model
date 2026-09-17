# Initial Extraction Failure and Repair

The first source audit completed all five fit recordings and exact canonical
position replay. It found 365 stationary windows, 31 agents and 45 contiguous
equal-position runs. No probe was trained and no development label was opened.

Saving row metadata failed with `TypeError: Object of type bool is not JSON
serializable`: source comparisons returned NumPy booleans. The initial registration
is retained as `configs/m3w_stationary_start_probe.json`; its bound source snapshot
is preserved locally under the failed output's `failed_source_snapshot` directory.
No prior identity is overwritten to simulate a successful resume.

The repair explicitly converts the two run-boundary flags to Python booleans.
A JSON serialization regression check covers the failing path. Diagnostic label,
features, folds, model settings and primary experiment remain unchanged. A new
v2 execution registration/output records the corrected implementation. Initial
extraction failure is not a failed forecasting result or a runtime training stall.
