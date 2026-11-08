# Scene Context Does Not Improve Exact Exchange Against Matched Controls

Time-aligned traffic produces a tiny positive comparison with fixed half,
but so does shuffled traffic. Neither planned direct contrast supports a
contribution from the correctly attributed motor histories.

[Protocol](protocol.md) | [Per-location paired metrics](analysis.json) |
[Updated main draft and supplement](../evidence_manuscript_v35/README.md)

84 fresh tree fits: four equal-width feature arms, seven held OpenDD
locations, three seeds. All edited forecasts preserve the fixed-half
recipient roster and actual query-total radius. All heads see the same
exact actions and label construction, with no change to the risk screen.
The original 2% evaluation and the independent roles are unchanged.

The input audit hash-verifies the original eight-timestamp context
caches and exactly aligns the pedestrian histories with the current
query roster. Motor histories are present at the last observed time for
5,132 of 10,475 targets. The features use positions and backward
differences, not released velocities or future label availability.
The test does not verify how the publisher's tracks were generated.

| Feature arm | Lower gain vs fixed half, pp [95% CI] | Exchanges | Original passes /606 |
| --- | ---: | ---: | ---: |
| Fixed half | Reference | 0 | 547 |
| Own only | +0.0008 [-0.0024, +0.0040] | 2,250 | 546 |
| Pedestrian context | +0.0006 [-0.0018, +0.0032] | 2,815 | 547 |
| Pedestrian + time-aligned motor | +0.0029 [+0.0011, +0.0050] | 2,800 | 547 |
| Pedestrian + shuffled motor | +0.0022 [+0.0009, +0.0036] | 2,748 | 547 |

The direct paired motor-vs-pedestrian difference is **-0.0007 pp**
[-0.0032, +0.0021], and motor-vs-shuffled **-0.0012 pp**
[-0.0035, +0.0008]. Differences of separate lower bounds against
fixed half are not pairwise lower bounds. The past-defined motor-supported
target slice also gives no established improvement vs pedestrian-only
(-0.0001 pp [-0.0017, +0.0017]). It is descriptive, with only two such
targets at the smallest site. All intervals use 3,000 location draws
after averaging seeds, not independent windows.

The motor arm recovers one original pass and loses one; all 30 undefined
views remain. Of 2,800 chosen exchanges, 1,280 are both beneficial and
nonincreasing in the actual risk numerators. The shuffled arm reaches
1,250 of 2,748. The primary contextual contribution criterion fails.
This does not rule out other interaction representations or imply
physical safety. Fixed half stays the development comparator.

The main paper now foregrounds matched motion, preservation of
easy-reference support, and the gap between hindsight and past-only
choice. Further development should test a genuinely different signal
or independent support-rich recordings rather than tune this result.
No new external outcomes, confirmation, deployment, Stage5C, or SMC.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_context_exchange.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_context_exchange.py run
.venv-pytorch/bin/python scripts/run_m3w_opendd_context_exchange.py replay
~~~

Native arm64 CPU, four Torch threads, zero workers. Fresh fitting and
scoring took 155.5 seconds. Heads checkpoint individually, and a
separate no-fit readout reproduces all results exactly. Thirty-five
targeted tests passed. No CREATE job was needed.
