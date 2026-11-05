# Initial Diagnostic Attempt

PID 3828 exited with code 1 before producing any result. The diagnostic
incorrectly resolved `direct` on the composed-runner module instead of its
decoupled-runner parent, causing AttributeError. No model was trained and no
new V/H prediction was evaluated. The original registration is retained;
registration_v2 binds the corrected diagnostic. Existing training, results,
roles and evaluation rules are unchanged.
