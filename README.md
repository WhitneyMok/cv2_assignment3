# 2D to 3D 

This repository contains source code written for the third and final assignment of a computer vision course. Following preceding assignments on depth-based 3D reconstruction and Structure from Motion (SfM), this one explores monocular 3D reconstruction using the Basel Face Model (BFM) [1]. BFM was built to fit 3D scans of 200 faces and resulted in a multidimensional 3D morphing function that, as also demonstrated in this work, captures knowledge about facial 3D geometry.

The [final report](https://github.com/WhitneyMok/cv2_assignment3/blob/master/docs/final_report_WhitneyMok_cv2assignment.pdf) also includes a comparison between this type of 3D reconstruction that exploits domain knowledge, dept-based reconstruction, and SfM. Detailed [problem statement and instructions](https://github.com/WhitneyMok/cv2_assignment3/blob/master/docs/Computer_Vision_2___Assignment_3__2D_to_3D.pdf) for all parts of the assignment are also available, and the correspondence between sections and code is as follows:  

Section 2 code: morpable_model.py

Section 3 code: pinhole_cam_model.py

Section 4 code: latentparamEstimation.py and latentparamEstimation_Model.py

Section 5 code: Texturing


All libraries required for this project can be installed with requirements.txt (into a virtual environment).
 
\
Author: WhitneyMok

Course: Computer Vision 2

University: University of Amsterdam

Date: May 31st, 2020



## References
[1] Pascal Paysan, Reinhard Knothe, Brian Amberg, Sami Romdhani, and Thomas Vetter. A 3d face model
for pose and illumination invariant face recognition. In 2009 Sixth IEEE International Conference on
Advanced Video and Signal Based Surveillance, pages 296{301. Ieee, 2009.
