# Method And Compute Reuse Notes

## Expert-Imbalance Control

Cortes,Mao,Mohri and Zhong's
[Optimized Deferral for Imbalanced Settings](https://proceedings.mlr.press/v306/cortes26b.html)
(ICML2026) gives MILD,a cost-sensitive expert-routing method with expert-specific
margins. The publisher-linked
[PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/cortes26b/cortes26b.pdf)
was read locally on1November2026,pages2-7including Sections3-5.3. Its Eq5 scales
score differences by the target expert's margin inside log-sum-exp,and Section5.3
weights these terms with continuous expert rewards. This is not one-hot best-
expert classification or an unqualified instruction to oversample oracle labels.

The existing TDEF application code is reusable for a current-bank comparator;
MILD would need a separately disclosed adapter with margins selected on legal
training/validation data. Neither expert imbalance nor a generic gain predictor
is itself a new contribution. No new MILD implementation or benchmark is claimed
here,and its guarantees are not transferred to our constrained integer choices,
partial native ADE labels or recording-level positive-harm ratios. This read is
not an audit of every appendix proof. Private PDF SHA256:
`e1e335bfa5de101a580197daf0bdbb4d6a0660c1a1852f4b6721fb095420ed72`.
The third-party PDF is excluded fromGit.

## Forecast-Seed Assets Already Available

The existing active-producer `models()` verifier was run read-only. All36model
receipt and private weight hashes pass,including actual4,000-update endpoints.
Nine history-arm models exist for single0/single1/single2and seeds17/29/43,
89,026parameters each,66,970,755checkpoint bytes combined. The producer freeze
SHA256is `cdafc57c63f810ead91bab69bc24392d6de828bce786d482eb0035e91b10e18a`.
Status:cached_verified weights,not new training,inference or scientific results.

The current registered cost trial intentionally still uses forecaster43only.
Existing17/29weights can support a prospectively registered forecast-seed
extension after a useful method is established;their availability does not
prove that the current three cost seeds are three full pipelines. New bank
features,cost fits and full readouts would still be required. No independent
roles are opened merely to inventory those source-trained assets.
