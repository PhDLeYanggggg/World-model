# Complete Source-Population Tables

Fresh aggregate calculation from hash-verified existing arrays, with raw
index/geometry replays. No new fits or deployment. All values are source
diagnostics on four previously explored sites, not independent testing.

## Population

| Site | Indexed | Complete | Partial | Absent | Static history |
| --- | ---: | ---: | ---: | ---: | ---: |
| coupa | 28,060 | 24,834 | 2,944 | 282 | 4,645 |
| deathCircle | 36,585 | 27,394 | 8,366 | 825 | 4,971 |
| gates | 20,722 | 15,281 | 4,955 | 486 | 2,594 |
| hyang | 90,389 | 76,409 | 12,774 | 1,206 | 8,154 |

All indexed queries are retained. Retrospective event labels require
all twelve futures; partial/absent future events are unknown. FDE requires
the twelfth target, not merely the last available target.

## Fixed Causal Baselines

Equal-site means. Pixel diagnostics are not metrically calibrated and do
not replace the registered past-normalized primary error.

### supported_masked

| Baseline | Normalized ADE | Normalized FDE | Pixel ADE diagnostic | Pixel FDE diagnostic |
| --- | ---: | ---: | ---: | ---: |
| constant_position | 115.027603 | 272.892608 | 53.902237 | 105.259938 |
| constant_velocity_causal_fd | 114.826067 | 272.514759 | 18.100943 | 39.464767 |
| damped_velocity_005 | 114.830555 | 272.534058 | 18.988278 | 43.096411 |
| damped_velocity_010 | 114.849780 | 272.591701 | 22.534813 | 53.550618 |
| damped_velocity_020 | 114.885947 | 272.681856 | 29.110735 | 69.613651 |
| constant_acceleration_causal | 115.830072 | 275.626377 | 72.309022 | 208.369692 |
| constant_turn_rate | 114.908230 | 272.727447 | 31.133334 | 73.738616 |

Rows: 172,957; final-label rows: 144,010.
Future-informed oracle headroom: 0.039520% versus CV.
This is not a learned or causal oracle.

### complete

| Baseline | Normalized ADE | Normalized FDE | Pixel ADE diagnostic | Pixel FDE diagnostic |
| --- | ---: | ---: | ---: | ---: |
| constant_position | 130.500691 | 273.072726 | 57.783909 | 105.241769 |
| constant_velocity_causal_fd | 130.281080 | 272.694517 | 19.513478 | 39.393075 |
| damped_velocity_005 | 130.286663 | 272.713932 | 20.598997 | 43.044021 |
| damped_velocity_010 | 130.308886 | 272.771663 | 24.653513 | 53.513495 |
| damped_velocity_020 | 130.349908 | 272.861901 | 32.016679 | 69.589467 |
| constant_acceleration_causal | 131.439020 | 275.806113 | 81.821894 | 208.266641 |
| constant_turn_rate | 130.374989 | 272.907395 | 34.271242 | 73.694419 |

Rows: 143,918; final-label rows: 143,918.
Future-informed oracle headroom: 0.038351% versus CV.
This is not a learned or causal oracle.

## Complete-Future Events

| Category | Windows | Scoped tracks | Disjoint spans | Normalized CV error share | Pixel CV error share | Oracle gain vs CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_stays | 8,566 | 453 | 860 | 0.000000% | 0.000000% | undefined: zero CV error |
| static_moves | 6,864 | 253 | 848 | 99.748062% | 0.665965% | -0.000000% |
| moving_stops | 14,848 | 635 | 1,736 | 0.046563% | 10.139256% | 24.185924% |
| moving_turns | 17,676 | 1,225 | 2,589 | 0.060521% | 24.647684% | 21.390621% |
| other_motion | 95,964 | 2,097 | 6,998 | 0.144855% | 64.547095% | 10.005680% |

Tracks and disjoint spans across categories are not additive independent
events. Exact float32 cached-label categories describe annotations, not
human-gold behavioral semantics. Tiny negative oracle differences at
approximately 1e-13 percent are floating-point summation roundoff.

## Site Consistency

| Site | Static-start windows | Normalized CV error share | Pixel CV error share | All oracle headroom |
| --- | ---: | ---: | ---: | ---: |
| coupa | 2,546 | 99.799568% | 1.716195% | 0.029696% |
| deathCircle | 987 | 99.746959% | 0.517333% | 0.040455% |
| gates | 420 | 99.662878% | 0.361125% | 0.051748% |
| hyang | 2,911 | 99.709645% | 0.560919% | 0.042699% |

## Other-Site Baseline Selection

For supported, complete and complete-moving cohorts, every held-source
fold selects causal CV using the other three source sites. This gives
zero gain over CV, not a new predictor. Complete-moving oracle headroom
is 15.216486%.

## Per-Recording Support

| Recording | Indexed | Complete | Partial | Absent | Static history |
| --- | ---: | ---: | ---: | ---: | ---: |
| coupa/video0 | 6,376 | 5,477 | 818 | 81 | 1,009 |
| coupa/video1 | 4,702 | 4,093 | 556 | 53 | 478 |
| coupa/video2 | 4,349 | 3,703 | 590 | 56 | 284 |
| coupa/video3 | 12,633 | 11,561 | 980 | 92 | 2,874 |
| deathCircle/video0 | 12,374 | 9,496 | 2,614 | 264 | 1,071 |
| deathCircle/video1 | 15,408 | 11,668 | 3,409 | 331 | 2,918 |
| deathCircle/video2 | 401 | 212 | 172 | 17 | 29 |
| deathCircle/video3 | 8,280 | 5,960 | 2,113 | 207 | 916 |
| deathCircle/video4 | 122 | 58 | 58 | 6 | 37 |
| gates/video0 | 2,012 | 1,539 | 434 | 39 | 92 |
| gates/video1 | 5,271 | 3,993 | 1,169 | 109 | 599 |
| gates/video2 | 2,906 | 2,237 | 612 | 57 | 540 |
| gates/video3 | 5,803 | 4,115 | 1,531 | 157 | 463 |
| gates/video4 | 1,541 | 1,100 | 402 | 39 | 271 |
| gates/video5 | 665 | 521 | 130 | 14 | 0 |
| gates/video6 | 419 | 313 | 95 | 11 | 12 |
| gates/video7 | 423 | 267 | 142 | 14 | 47 |
| gates/video8 | 1,682 | 1,196 | 440 | 46 | 570 |
| hyang/video0 | 12,328 | 10,423 | 1,742 | 163 | 1,118 |
| hyang/video1 | 8,544 | 7,454 | 996 | 94 | 643 |
| hyang/video10 | 4,251 | 3,648 | 551 | 52 | 492 |
| hyang/video11 | 6,982 | 5,479 | 1,371 | 132 | 816 |
| hyang/video12 | 1,899 | 1,424 | 433 | 42 | 52 |
| hyang/video13 | 1,236 | 884 | 319 | 33 | 0 |
| hyang/video14 | 1,482 | 1,174 | 281 | 27 | 268 |
| hyang/video2 | 10,035 | 8,885 | 1,051 | 99 | 1,085 |
| hyang/video3 | 8,143 | 6,526 | 1,479 | 138 | 964 |
| hyang/video4 | 20,241 | 17,620 | 2,397 | 224 | 1,027 |
| hyang/video5 | 7,751 | 6,714 | 949 | 88 | 526 |
| hyang/video6 | 6,347 | 5,430 | 838 | 79 | 846 |
| hyang/video7 | 896 | 591 | 279 | 26 | 265 |
| hyang/video8 | 236 | 152 | 77 | 7 | 52 |
| hyang/video9 | 18 | 5 | 11 | 2 | 0 |

## Verification

- 198 array hashes checked; 175,756 raw past-index keys rebuilt.
- 1,230,292 baseline/query ADE and FDE pairs independently reduced.
- 256 exact raw geometry/label replays; 33 future-poison input checks.
- Completed audit rerun exactly matches every saved row and aggregate.
- No main, bookstore, original validation/test or external readout.
- No primary metric change, new training, Stage5C or SMC.
