# Fixed detector-resolution diagnostic

The first-frame bridge retains all eight early frames and shows zero
person links with the existing 4 by 3 grid. Test one fixed 8 by 5 grid
with 20% overlap on the same prefix, without selecting crops from labels.
Weights, model resize, class list, score 0.35, classwise NMS 0.5,
causal tracker, bridge acceptance, nominal label clock and matching
radius all stay fixed. Run independently a second time.

Compare detections and links on identical geometry and label support.
A positive difference diagnoses scale sensitivity in this prefix, not
general detection accuracy, validated metric coordinates, or forecast
gain. No offset fitting, new scientific role or confirmation access.
