# Stage42 Experiment Tables

| experiment | source | all | t50 | t100 | hard | easy | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stage42-B protected endpoint/external validation | fresh_run | 0.2103 | 0.1365 | 0.1469 | 0.2038 | -0.1451 | protected composite-tail endpoint dynamics; external source-fold eval |
| Stage42-C protected full-waypoint dynamics ADE | fresh_run | 0.1858 | 0.1480 | 0.2286 | 0.1952 | -0.0000 | actual reconstructed future waypoint labels; positive on ETH_UCY and TrajNet |
| Stage42-C protected full-waypoint dynamics FDE | fresh_run | 0.1938 | 0.2158 | n/a | n/a | n/a | full-waypoint FDE summary |
| Stage42-E best deployable safety-floor policy | fresh_run | 0.2103 | 0.1365 | 0.1469 | 0.2038 | 0.0000 | teacher_floor_required_for_current_deployment |
