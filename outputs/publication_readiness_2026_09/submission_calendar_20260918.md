# CVPR 2027 Submission Evidence and Working Calendar

## Material Passport

Fresh official-web check on2026-09-18, not acceptance or readiness evidence.
Rechecked during the matched motion-loss run; this calendar does not replace
missing results. The official CFP still lists the dates below, and the direct
2027 AuthorGuidelines path still returned404. No registration was submitted.

Rechecked again on2026-09-20 during the frozen-predictor coupling run: the
[official dates](https://cvpr.thecvf.com/Conferences/2027/Dates) still list
Nov10/16/23 registration/paper/supplement deadlines. The linked2027 author guide
again returned404 and LLM-use details remain pending. The
[CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) also describes work
appearing after2026-09-15 as generally contemporaneous, but still requires credit
and discussion; influenced prior work cannot be omitted. No change to the
internal targets below, and no claim that format compliance is complete.

## Official Dates and Current Policy Availability

2026-09-21 live recheck: the official dates remain Nov10/16/23 (AoE). The
linked2027 AuthorGuidelines still returns404; LLM-use details remain unfinished.
This verifies dates and the availability gap, not formatting compliance. Current
research priorities below are refreshed against the completed native-loss and
bounded-cost studies rather than the older normalized-loss results.

The official dates page lists registration on2026-11-10, full submission
on2026-11-16 and supplementary materials on2026-11-23, all Anywhere on Earth.
Reviews are scheduled for2027-01-25 and decisions for2027-02-25.
[Official dates](https://cvpr.thecvf.com/Conferences/2027/Dates).

The call requires current OpenReview profiles for all authors; it warns that
new profiles without institutional email may require up to two weeks of
moderation. It states a dual-submission restriction on substantially similar
work during2026-11-16 to2027-02-22. Its LLM-use details remain under preparation.
[Official CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers).

The CFP's Author Guidelines link returned404 during both direct opening and
link-following. Consequently this check does **not** verify the2027 page limit,
template version, anonymization specifics, supplement size or final AI-use
rules. Do not silently substitute last year's policies. Recheck the official
guidelines before format freeze. No paper or author registration is submitted.

## Internal Working Targets, Not Conference Requirements

| Internal target | Evidence required before proceeding |
| --- | --- |
| Sep30 | Resolve whether an admissible candidate predictor has useful gain; retain failed controls and loss decomposition. |
| Oct10 | Matched independent-agent versus joint-scene intervention evidence; explicit go/no-go on the proposed contribution. |
| Oct20 | Freeze selection only if supported; independent calibration/confirmation eligibility and scientific decisions must already be resolved. Never relabel exposed data. |
| Oct27 | Complete evidence-limited English draft, figures, related work and limitations. |
| Nov03 | Reproduction review, anonymous artifact audit and author-account readiness check. Public project history does not make the submission artifact anonymous. |
| Nov09 | Internal registration buffer; user confirms any actual registration/submission. |
| Nov12 | Internal main-paper freeze, policy/template check and final author review, leaving four days to the stated submission date. |
| Nov20 | Internal supplement freeze and reproduction checks, three days before the stated supplement date. |

This is a conditional schedule, not authorization to inspect sealed outcomes or
weaken the protocol to meet a date. If candidate gains, scene-level risk evidence
or independent confirmation remain absent, report the shortfall rather than
call the paper ready. The user retains the final submission decision.

## Current Priority Gaps

### Latest Status, 2026-09-21

The native-coordinate loss comparison now supplies a useful predictor:
7.63% source-development ADE gain over CV, but exact-zero protection fails.
Joint versus unary controls are complete and make identical decisions; coupling
is no longer presented as a supported main contribution. The bounded-cost
primary arm improves average accuracy but fails protection. A fixed secondary
fraction-loss arm gives2.44% source ADE gain and preserves observed complete
exact-CV cases, with unresolved incomplete outcomes and conditional underharm.
All four source sites remain design-exposed. These developments supersede the
older "no candidate gain" diagnosis below, not the historical results themselves.

The current critical path is: matched native-loss public comparator; conditional
harm reliability; eligible new physical scenes and approved calibration/confirmation
roles; then a fixed independent readout and complete paper claims. Twelve EqMotion
K=1 native-loss fits are newly registered and training; they cannot be counted as
completed evidence here. No default larger-model sweep, closed-label tuning or
weaker safety criterion is authorized by the calendar. Local pilot cost is about
39s/100updates; the full registered matrix is estimated at5-6hours. CREATE is not
needed for this bounded local run; no new remote asset or scheduler state is claimed.

### Historical Priority Snapshot

1. Candidate dynamics: the 12-model site-crossfit, matched static-loss repair,
   36-head pretrained comparison, 24-head centering comparison and 24-head
   episode-sampling comparison are complete. None yields positive excluded-site
   forecasts. Equal-episode sampling worsens geometry/centered gains to
   -37.327%/-54.992% and changes the expected training risk. A further24-head
   importance-correction experiment repairs that large harm but still gives
   -0.032%/-0.275% versus CV. The fresh event audit finds only 55 annotation
   groups/47 scoped tracks among 207 half-box-excursion windows; independent
   support and predictive past information remain unresolved.
2. Conditional gain/harm: lower training jitter must not be mistaken for useful
   movement prediction. A candidate with measurable benefit is required before
   arguing that joint intervention solves a real prediction problem.
3. Independent evidence: matched public forecasting/deferral controls, useful
   joint-versus-independent contrasts, independent calibration/confirmation,
   source/scene uncertainty and anonymous reproduction remain incomplete.
4. Resources and policy: the latest 240k-update small-head comparison completed
   locally in 674 summed fitting seconds with exact resume. This does not predict
   end-to-end encoder cost. CREATE asset and scheduler state remain unverified;
   the prior SSH access failure has not been repaired in this comparison. Final
   2027 formatting/AI-use policy remains unverified. Updating these experiment
   gaps is not a new official-web date or policy check.

No metric/seconds/true3D/foundation claim, no Stage5C execution orSMC. Meeting an
internal date is not a research gate pass.
