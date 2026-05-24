# Stage38 Project World Model Gap

- Stage38 freezes Stage37 and confirms UCY external robustness, but ETH/TrajNet held-out tests remain blockers.
- Bounded correction is trained but not deployed because it does not beat Stage37 safely on all/hard while preserving t50/easy.
- Current best external model remains Stage37 selector.
- The project is still dataset-local 2.5D, not metric/seconds/3D/foundation.
- Shortest next path: rebuild external held-out splits for ETH/TrajNet, then train correction on domains with verified t50/hard headroom and validate per-domain bootstrap.
