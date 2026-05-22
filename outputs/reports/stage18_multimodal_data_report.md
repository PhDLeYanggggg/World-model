# Stage 18 Multimodal Data Report

- No gated/license-restricted data was downloaded.
- Existing local derived episodes are used for quick JEPA construction.
- existing episode paths: `1078`
- multimodal/raster-ready datasets: `5`

| dataset | local path found | multimodal ready | verified t50 | verified t100 | next action |
| --- | --- | --- | --- | --- | --- |
| aerialmpt | True | True | False | False | none for quick JEPA; provide raw SDD/OpenTraj for stronger multimodal training |
| eth_ucy | True | True | False | False | none for quick JEPA; provide raw SDD/OpenTraj for stronger multimodal training |
| eth_ucy_ewap_stage14 | True | True | True | True | none for quick JEPA; provide raw SDD/OpenTraj for stronger multimodal training |
| eth_ucy_ewap_stage15 | True | True | True | True | none for quick JEPA; provide raw SDD/OpenTraj for stronger multimodal training |
| trajnet | True | True | False | False | none for quick JEPA; provide raw SDD/OpenTraj for stronger multimodal training |
| Stanford Drone Dataset | False | False | False | False | provide local path after accepting license |
| OpenTraj-supported datasets | False | False | False | False | provide local path after accepting license |
| full TrajNet++ | True | False | False | False | run verifier/converter |
| full ETH/UCY | True | False | False | False | run verifier/converter |
| AerialMPT longer sequences | True | False | False | False | run verifier/converter |
