# Stage36 Project World Model Gap

- Stage36 isolates the remaining blocker to t+50 transfer.
- There is t+50 oracle headroom, but horizon-specific models cannot capture it safely on held-out external scenes.
- The likely gap is missing causal full-history interaction/curvature/TTC and weak train-only goal context for held-out UCY scenes.
- Next: rebuild external feature rows from full causal windows, not just current/past-start geometry; then train per-dataset t+50 policies with held-out scene validation.
