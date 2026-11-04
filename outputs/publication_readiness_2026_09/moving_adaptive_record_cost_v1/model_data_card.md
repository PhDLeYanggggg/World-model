# Model And Data Card

CostHead width64,28229parameters,428causal input fields (380local+48zeros). Original Cfit-only normalizers, seeds17/29/43,2000updates per head, CPU4/interop1/workers0. Record IDs and current detached actions are training-loss metadata, not network inputs. Future endpoint/mask/easy label never enters inference. Frozen9EqMotion trajectory proposals and physical floor are unchanged. No scene-image or JEPA training.

195286past-eligible SOURCE rows,64admitted recordings;23supported source groups. Original P/C/Hsource exclusions remain. Repeated/overlapping windows are not IID samples. Partial labels retain conservative bounds; unavailable full-roster FDE stays NA. No independent confirmation, public cold-start reproduction proof, deployment, Stage5C or SMC.
