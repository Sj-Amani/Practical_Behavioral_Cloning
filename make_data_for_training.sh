#!/bin/sh

# directory of the behavioral cloning package on your local pc/laptop/AWS 
dir_workspace=/home/s-amani/Desktop/Resesarch/Udacity/Self_Driving_Car_Engineer_Nanodegree/03-Projects/P04/Practical_Behavioral_Cloning  		# the directory where this package is in it
# directories of the camera images for left/middle/right driving
dir_left_side_driving=/home/s-amani/Desktop/Resesarch/Udacity/Self_Driving_Car_Engineer_Nanodegree/03-Projects/P04/Practical_Behavioral_Cloning/left		# the directory to the recorded left side driving data
dir_middle_side_driving=/home/s-amani/Desktop/Resesarch/Udacity/Self_Driving_Car_Engineer_Nanodegree/03-Projects/P04/Practical_Behavioral_Cloning/middle		# the directory to the recorded middle side driving data
dir_right_side_driving=/home/s-amani/Desktop/Resesarch/Udacity/Self_Driving_Car_Engineer_Nanodegree/03-Projects/P04/Practical_Behavioral_Cloning/right		# the directory to the recorded right side driving data


cd $dir_middle_side_driving
python $dir_workspace/horizontal_flip.py
cd $dir_left_side_driving
python $dir_workspace/horizontal_flip.py
cd $dir_right_side_driving
python $dir_workspace/horizontal_flip.py

cd $dir_workspace
mkdir data
cd data
cat $dir_left_side_driving/driving_log.csv > driving_log_left.csv
cat $dir_left_side_driving/driving_log_flipped.csv >> driving_log_left.csv

cat $dir_middle_side_driving/driving_log.csv > driving_log_middle.csv
cat $dir_middle_side_driving/driving_log_flipped.csv >> driving_log_middle.csv

cat $dir_right_side_driving/driving_log.csv > driving_log_right.csv
cat $dir_right_side_driving/driving_log_flipped.csv >> driving_log_right.csv

python $dir_workspace/map_driving_data.py

cd $dir_workspace
python middle_driving_data_preprocessing.py
python left_right_driving_data_preprocessing.py
python merge_all_preprocessed_data.py

echo "Complete the preprocessing"
echo "For training the model, do this:"
echo "python model.py" 

echo "Have fun!"
