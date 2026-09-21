# Real Training Loss and Sampling Coverage

All registered endpoints are verified before this reduction. No held predictions are read.
The trace contains sampled training minibatches, not epoch loss or validation performance.
Old and native objectives assign different sample weights; their loss values are not directly comparable.
Full-population sampling means every eligible training row can be drawn, not that every row was drawn.

Completed fits: 24; optimizer updates: 96,000; row draws: 6,144,000.
Summed fit time: 33.40 minutes, excluding setup and later evaluation.

| Held Site | Objective | Seed | Unique / Eligible Training Rows | Coverage (%) | Early / Late Logged Batch Loss |
| --- | --- | ---: | --- | ---: | --- |
| coupa | past_normalized | 17 | 108,465 / 147,696 | 73.44 | 0.89954 / 0.66984 |
| coupa | past_normalized | 29 | 108,618 / 147,696 | 73.54 | 1.47351 / 0.66027 |
| coupa | past_normalized | 43 | 108,359 / 147,696 | 73.37 | 1.02848 / 0.99465 |
| coupa | native_coordinate | 17 | 108,465 / 147,696 | 73.44 | 0.88467 / 0.99685 |
| coupa | native_coordinate | 29 | 108,618 / 147,696 | 73.54 | 0.93565 / 0.93489 |
| coupa | native_coordinate | 43 | 108,359 / 147,696 | 73.37 | 0.95631 / 0.94311 |
| deathCircle | past_normalized | 17 | 102,209 / 139,171 | 73.44 | 0.61653 / 0.80760 |
| deathCircle | past_normalized | 29 | 102,324 / 139,171 | 73.52 | 0.76438 / 0.49481 |
| deathCircle | past_normalized | 43 | 102,083 / 139,171 | 73.35 | 0.98043 / 0.55378 |
| deathCircle | native_coordinate | 17 | 102,209 / 139,171 | 73.44 | 0.97976 / 0.94865 |
| deathCircle | native_coordinate | 29 | 102,324 / 139,171 | 73.52 | 0.96816 / 1.02397 |
| deathCircle | native_coordinate | 43 | 102,083 / 139,171 | 73.35 | 0.97360 / 0.99885 |
| gates | past_normalized | 17 | 114,868 / 155,034 | 74.09 | 0.78916 / 0.51618 |
| gates | past_normalized | 29 | 114,960 / 155,034 | 74.15 | 1.22564 / 0.43848 |
| gates | past_normalized | 43 | 114,770 / 155,034 | 74.03 | 0.40261 / 0.59255 |
| gates | native_coordinate | 17 | 114,868 / 155,034 | 74.09 | 0.99094 / 0.87440 |
| gates | native_coordinate | 29 | 114,960 / 155,034 | 74.15 | 1.04575 / 0.95689 |
| gates | native_coordinate | 43 | 114,770 / 155,034 | 74.03 | 1.00455 / 0.95080 |
| hyang | past_normalized | 17 | 80,115 / 85,367 | 93.85 | 0.81083 / 1.10398 |
| hyang | past_normalized | 29 | 80,094 / 85,367 | 93.82 | 1.04457 / 0.51074 |
| hyang | past_normalized | 43 | 80,173 / 85,367 | 93.92 | 0.49319 / 0.63746 |
| hyang | native_coordinate | 17 | 80,115 / 85,367 | 93.85 | 0.98489 / 0.87395 |
| hyang | native_coordinate | 29 | 80,094 / 85,367 | 93.82 | 1.02764 / 0.89941 |
| hyang | native_coordinate | 43 | 80,173 / 85,367 | 93.92 | 1.01653 / 0.97459 |

Early/late values average the first/last ten logged minibatches. They are not evidence of held-scene improvement.
Across-fit draw counts are not independent sample counts. All training held-site draw counts are zero.
Checkpoint/sampler replay and final held-source metrics are separate verification steps.
