# 2D to 3D: 3rd assignment of Computer Vision 2 

This repository contains the source code I wrote for the final, individual assignment of a computer vision course. Following preceding assignments on depth-based 3D reconstruction and Structure from Motion (SfM), this one explores monocular 3D reconstruction using the Basel Face Model (BFM) [1]. BFM was built to fit 3D scans of 200 faces and resulted in a multidimensional 3D morphing function that, as also demonstrated in this work, captures knowledge about facial 3D geometry.

The final report (in */docs*) also includes a comparison between this type of 3D reconstruction that exploits domain knowledge, dept-based reconstruction, and SfM. 
That folder also contains detailed instructions for all sections of this assignment.  

The correspondence between sections and code is as follows:  

Section 2 code: morpable_model.py

Section 3 code: pinhole_cam_model.py

Section 4 code: latentparamEstimation.py and latentparamEstimation_Model.py

Section 5 code: Texturing


All libraries required for this project can be installed with requirements.txt

NB: The location of unzipped *Data* folder should not be changed (i.e. the folder needs to stay in root). *Data* contains BFM and used ground truth image for section 4 and 5.

\
Author: WhitneyMok

Course: Computer Vision 2

University: University of Amsterdam



##References
[1] Pascal Paysan, Reinhard Knothe, Brian Amberg, Sami Romdhani, and Thomas Vetter. A 3d face model
for pose and illumination invariant face recognition. In 2009 Sixth IEEE International Conference on
Advanced Video and Signal Based Surveillance, pages 296{301. Ieee, 2009.