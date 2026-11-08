# Causal first-frame bridge

The direct seasonal image registration failed some frames in the
early prefix. Compare a fixed bridge: register the first raw frame to
the map, then register each current frame to that already observed
first frame using static image matches. Do not use later images or
trajectory coordinates to estimate the bridge. Exclude the same
holdout patches from both image fits.

Keep the same held map patch acceptance rule (three patches with
correlation above 0.30 and shifts at most six map pixels). On each
raw-to-first fit additionally require 40 inliers, four occupied
quadrants, hull support at least 25%, and a 95th percentile residual
at most two raw working-grid pixels. Check the same frames and the
same fixed prefix association. No hypothesis or safety threshold is
selected using later forecast outcomes.
