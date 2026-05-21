# Stage 5B.5 Problem Statement

Stage 5B.5 is not a latent generative stage. It is a deterministic learned dynamics repair stage.

Stage 5B showed that simple causal baselines can dominate smooth real trajectories, especially TGSIM traffic. A learned residual that only adds small corrections is not enough if the benchmark is mostly straight, smooth, or near-inertial. The core question is whether a deterministic temporal-interaction model can beat the strongest causal baseline on hard subsets: turning, acceleration/deceleration, stop/go, close interaction, near collision, high density, and long-horizon nonlinear motion.

Current hard constraint: latent generative modeling and SMC remain disabled until deterministic dynamics passes the hard gates.
