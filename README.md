STFprediction
=============

In order to use the scripts:

(0) Download this repository
 
(1) Install Deep Neural Network scripts from http://www.robots.ox.ac.uk/~vgg/software/deep_eval/
    
    * you might interesting in this link https://software.intel.com/en-us/intel-education-offerings, which allows you to download and install MKL free academic licence

(2). Put extractCNN.m from this repository in the deep eval installation folder.

(3) Modify path at the working_folder variable to point to the images folder path.

(4) Run predict.py script in terminal with the flag --f to point to the dataset directory.

the command will look like
  >>python predict.py --f=/home/localadmin/dataset


(5) Wait until it done. The results will be in results/ folder in the dataset directory 

*the dataset folder should have images/ folder that stores all the input images you want to do segmentation.
