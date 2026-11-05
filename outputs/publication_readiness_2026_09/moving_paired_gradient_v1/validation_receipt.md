# Verification Receipt

Scientific analysis SHA256:
`b2203fbe9c0bb77abe46fa483f9b3b4d2b09b8258fca59375c9e59f13548703c`.
The original and revised verification receipts bind exactly the same result.

- Initial audit PID22551:46.317844875s, peak RSS6456229888bytes.
- Original exact verification PID22887:46.259775917s, peak6455902208bytes.
- Corrected-test wrapper verification PID23156: same numerical result.
- Completed replay PID23372:47.097485625s, peak6489702400bytes;
  all nine then-existing public files unchanged. Exit0 observed.
- Twenty-six combined focused cases pass in2.50s, including paired gradients,
  neighbor diagnostics, paired objectives, loss geometry, original joint loss,
  discrimination and completed-replay guards.
- The additional neighbor replay-receipt case passes in1.36s:27 distinct cases.
- Full historical suite not rerun. No new held data or model checkpoint.

The initial mask-mutation test failed because the provenance guard worked.
The corrected test explicitly requires that rejection and independently checks
zero unknown-label pair contributions. No scientific evaluator change was made.
Original registration, validation revision and failed-check explanation remain.

The original projection-training screen fails. Reproducible arithmetic does
not establish downstream utility, independent confirmation or a safety pass.
