# Severity Auxiliary Results

## Material Passport
144 fresh Torch fits /288,000 updates. Source forecasts and matched controls are cached_verified.
All six assignments, three seeds and four held localities; source development only.
Harm-weighted auxiliary output is not ordinary membership probability and is never multiplied into costs.

| Comparison / pair / assignment | Easy-harm MSE gain % | 95% locality CI |
|---|---:|---:|
| severity_vs_original / full / producer0_controller1 | -0.5867588303121648 | [-1.0018225210047302, -0.17169513961959937] |
| severity_vs_original / full / producer0_controller2 | -19.80488251235358 | [-42.93709321244083, 0.10545565290607638] |
| severity_vs_original / full / producer1_controller0 | 2.3801381896312805 | [-1.116822936423262, 8.107991294074647] |
| severity_vs_original / full / producer1_controller2 | -5.184266097623999 | [-15.004801186097293, 0.13094777767066124] |
| severity_vs_original / full / producer2_controller0 | -0.45491837628811993 | [-3.576068512588278, 2.573361093113135] |
| severity_vs_original / full / producer2_controller1 | -20.140145371784957 | [-48.71367635320059, 0.7917569774139195] |
| severity_vs_original / motion_only / producer0_controller1 | -7.896000648394203 | [-23.181358799421663, -0.16151096568830603] |
| severity_vs_original / motion_only / producer0_controller2 | -48.69343202585527 | [-100.58062247623752, 0.15556933315645577] |
| severity_vs_original / motion_only / producer1_controller0 | -7.160298604112335 | [-20.183967474264378, 0.07795473151897081] |
| severity_vs_original / motion_only / producer1_controller2 | -12.83353872873443 | [-38.19714096303725, 0.024237094858626053] |
| severity_vs_original / motion_only / producer2_controller0 | -0.5299107580957974 | [-6.417980179191433, 5.719126512865005] |
| severity_vs_original / motion_only / producer2_controller1 | 6.1771439470018095 | [-0.2662387569917385, 17.014768261962647] |
| severity_vs_control / full / producer0_controller1 | -0.5867588303121648 | [-1.0018225210047302, -0.17169513961959937] |
| severity_vs_control / full / producer0_controller2 | -19.80488251235358 | [-42.93709321244083, 0.10545565290607638] |
| severity_vs_control / full / producer1_controller0 | 2.3801381896312805 | [-1.116822936423262, 8.107991294074647] |
| severity_vs_control / full / producer1_controller2 | -5.184266097623999 | [-15.004801186097293, 0.13094777767066124] |
| severity_vs_control / full / producer2_controller0 | -0.45491837628811993 | [-3.576068512588278, 2.573361093113135] |
| severity_vs_control / full / producer2_controller1 | -20.140145371784957 | [-48.71367635320059, 0.7917569774139195] |
| severity_vs_control / motion_only / producer0_controller1 | -7.896000648394203 | [-23.181358799421663, -0.16151096568830603] |
| severity_vs_control / motion_only / producer0_controller2 | -48.69343202585527 | [-100.58062247623752, 0.15556933315645577] |
| severity_vs_control / motion_only / producer1_controller0 | -7.160298604112335 | [-20.183967474264378, 0.07795473151897081] |
| severity_vs_control / motion_only / producer1_controller2 | -12.83353872873443 | [-38.19714096303725, 0.024237094858626053] |
| severity_vs_control / motion_only / producer2_controller0 | -0.5299107580957974 | [-6.417980179191433, 5.719126512865005] |
| severity_vs_control / motion_only / producer2_controller1 | 6.1771439470018095 | [-0.2662387569917385, 17.014768261962647] |
| severity_vs_ordinary / full / producer0_controller1 | -0.9953351550914904 | [-2.1150706614607575, 0.0002487239591399959] |
| severity_vs_ordinary / full / producer0_controller2 | -11.565279938578596 | [-25.70218064947609, 0.21977844787433137] |
| severity_vs_ordinary / full / producer1_controller0 | 1.0416342892893546 | [-1.8856956133404408, 4.996329417293868] |
| severity_vs_ordinary / full / producer1_controller2 | -3.8195510962069523 | [-13.412005597611442, 1.8993103673090888] |
| severity_vs_ordinary / full / producer2_controller0 | -0.9803887061302844 | [-4.466961503798244, 3.0471025956939117] |
| severity_vs_ordinary / full / producer2_controller1 | -10.697320258982318 | [-35.05849432970865, 3.207669406129577] |
| severity_vs_ordinary / motion_only / producer0_controller1 | -11.430001445041484 | [-31.649945701772857, -0.30735587448452323] |
| severity_vs_ordinary / motion_only / producer0_controller2 | -51.33050056974781 | [-120.88572401228896, 0.07951423104491354] |
| severity_vs_ordinary / motion_only / producer1_controller0 | -14.144052615264888 | [-37.89786003776437, -0.025106923196898154] |
| severity_vs_ordinary / motion_only / producer1_controller2 | -4.137893912249105 | [-12.647369608116986, 0.2037738374654263] |
| severity_vs_ordinary / motion_only / producer2_controller0 | 4.6744516469602075 | [-0.714308429455859, 13.412325907270759] |
| severity_vs_ordinary / motion_only / producer2_controller1 | 9.483129997567346 | [-0.2583748167980604, 19.224634811932752] |

Three seeds averaged within locality, then3,000 resamples of four localities. Roles/windows dependent; exploratory and not multiplicity-adjusted.
The original forecaster, denominator readout, sampling and deployment remain frozen. No independent roles opened.
No metric/seconds, human-gold, true3D, foundation, physical-safety, independent-confirmation or deployment claim. Stage5C/SMC off.

```json
{
  "primary_six_positive_vs_all_controls": false,
  "tail_coverage_all_harm_guards": false,
  "development_cost_signal": false,
  "new_policy_evaluated": false,
  "independent_confirmation": false,
  "deployment_changed": false,
  "submission_ready": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```
