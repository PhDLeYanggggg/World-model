# Stage41 Fixed Policy Confirmation Audit

- source: `fresh_run`
- deployment decision: `candidate_needs_fresh_external_confirmation_before_deployment`
- stage37 margin pass: `True`
- stress pass: `True`
- fresh confirmation pass: `False`
- max domain easy degradation: `0.0055509018258728116`
- split overlap audit: `{'row_overlap': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'source_file_overlap': {'train_val': [], 'train_test': [], 'val_test': []}, 'row_overlap_pass': True, 'source_file_overlap_pass': True}`
- caveat: `This audit freezes the post-diagnostic policy and stress-tests it by domain/source/scene, but it does not create a new external dataset or independent locked test. It is not final deployable proof.`
