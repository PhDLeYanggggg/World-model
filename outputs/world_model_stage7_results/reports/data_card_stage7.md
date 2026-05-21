# Stage 7 Data Card

Stage 7 adds inferred scene packs and candidate goals. Inferred goals are not true observed goals.

## Scene Data Audit
| dataset | local | unit | homography | t50 | t100 | metric_eval |
| --- | --- | --- | --- | --- | --- | --- |
| stanford_drone_dataset | False | pixel_or_image_coordinate_until_homography | False | False | False | False |
| opentraj_supported_pedestrian | False | varies | False | False | False | False |
| full_trajnetplusplus | True | dataset_coordinate | False | False | False | False |
| full_eth_ucy | True | dataset_coordinate_or_meter_depending_source | False | False | False | False |
| aerialmpt_long | True | pixel | False | False | False | False |

## Scene Packs
| dataset | scene | goals | walkable | goal_source |
| --- | --- | --- | --- | --- |
| eth_ucy | eth_ucy_biwi_hotel | 4 | inferred_bbox_not_manual | inferred_scene_goal_from_training_endpoints |
| tgsim | tgsim_foggy_bottom | 5 | inferred_bbox_not_manual | inferred_scene_goal_from_training_endpoints |
| tgsim_i90 | tgsim_i90 | 4 | inferred_bbox_not_manual | inferred_scene_goal_from_training_endpoints |
| trajnet | trajnet_bookstore_0 | 4 | inferred_bbox_not_manual | inferred_scene_goal_from_training_endpoints |

## GoalBench
| dataset | episodes | candidate_goal_count_mean | goal_label_entropy | majority_goal_label | top1_majority_baseline | top3_majority_baseline | horizons | goal_ambiguity_score | goal_prediction_meaningful |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eth_ucy | 23 | 4.0 | 1.0844458991370884 | 0 | 0.6086956521739131 | 0.9130434782608695 | [10] | 0.7822623603987253 | True |
| tgsim | 32 | 5.0 | 1.479322678324495 | 0 | 0.34375 | 0.8125 | [100] | 0.9191548595292999 | True |
| tgsim_i90 | 31 | 4.0 | 1.279145601809765 | 0 | 0.3548387096774194 | 0.9032258064516129 | [100] | 0.9227085081529384 | True |
| trajnet | 32 | 4.0 | 1.3169718565075785 | 1 | 0.375 | 0.875 | [10] | 0.9499943831869075 | True |

