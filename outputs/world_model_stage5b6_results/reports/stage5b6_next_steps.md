# Stage 5B.6 Next Steps

1. Add a legal real pedestrian/drone source with verified t+50/t+100, preferably SDD after license acceptance or full OpenTraj/ETH-UCY with longer raw tracks.
2. Build multi-agent episodes instead of single-primary-agent windows so the interaction encoder influences actual trajectory prediction, not only diagnostic features.
3. Increase hard subset size to at least 50 episodes per official hard gate and retrain the gated residual with reliable hard validation.
