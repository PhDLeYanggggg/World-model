+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++ AerialMPT: A Dataset for Pedestrian Tracking in Aerial Imagery +++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


+++ Dataset Folder Structure ++++++++++++++++++++++++++++++++++

The train and test folders contain the sequences as sub-folders.
Each sequence folder contains
- images
- <sequence name>_gts.txt: annotations in the MOT format (see https://motchallenge.net/instructions/): 
	Format:	frame_ID,tracking_ID,x1,y1,w,h,1,-1,-1
- image_list.txt
- <sequence name>.mp4: track visualization


+++ Reference +++++++++++++++++++++++++++++++++++++++++++++++++

If you find AerialMPT useful in your work, please cite:

@INPROCEEDINGS{aerialmpt,
  author={Kraus, Maximilian and Azimi, Seyed Majid and Ercelik, Emec and Bahmanyar, Reza and Reinartz, Peter and Knoll, Alois},
  booktitle={25th International Conference on Pattern Recognition (ICPR)}, 
  title={AerialMPTNet: Multi-Pedestrian Tracking in Aerial Imagery Using Temporal and Graphical Features}, 
  year={2021},
  pages={2454-2461},
}



+++ Licence +++++++++++++++++++++++++++++++++++++++++++++++++++

The AerialMPT dataset is relased under CC BY-SA 4.0. For details, please see https://creativecommons.org/licenses/by-sa/4.0/.