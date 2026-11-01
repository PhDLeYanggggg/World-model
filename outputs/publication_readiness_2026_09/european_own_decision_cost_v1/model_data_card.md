# Cost Model And Data Card

36real warm-started Torch cost heads,31,119active parameters each. No new trajectory or latent generator.
Shared differentiable decoder:causal available-candidate R/R_easy mean,B capped at R,H/H_easy retained.
Input:462frozen causal fields and original causal cost/envelope/initial-choice context. Future labels enter losses/evaluation only.
Two objectives differ only in fixed versus detached current native choices used by the query-risk residual.
Training targets are pointwise known-action costs and complete nonempty selected-query residuals. Unknown targets stay unknown.
EuropeanSquares exposed SOURCE,149,510targets,74recordings,12localities,obs8/pred12/rawstride12.
Oblique detector-silver/local-image,not metric,seconds,human gold,true3D or foundation.
Three cost seeds share one forecaster. No independent selection/calibration/confirmation or whole-pipeline OOF claim.
Every original2%risk and undefined-support rule stays unchanged. Predicted feasibility is not physical safety.
Raw data,caches,features,weights and third-party media remain outsideGit. Public scalar evidence is reproducible separately.
