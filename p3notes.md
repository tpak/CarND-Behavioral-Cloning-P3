# **Behavioral Cloning Project Notes**

## These are really just notes to myself

[//]: # (Image References)
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"


**Behavioral Cloning Project**
this is a quick note to myself

Udacity github repo
https://github.com/udacity/CarND-Behavioral-Cloning-P3

Sample training data
https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

My training data:
https://s3.amazonaws.com/tpak-carnd/CarNDTrackData.zip


Project rubric
https://review.udacity.com/#!/rubrics/432/view

Nvidia paper:
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

Comma.ai model:
https://github.com/commaai/research/blob/master/train_steering_model.py

---
my data 

~/projects/CarND-Behavioral-Cloning-P3/CarNDTrackData2 master*
carnd-term1 ❯ cat ../CarNDTrackData/leftccw/driving_log.csv | wc -l
    1997

~/projects/CarND-Behavioral-Cloning-P3/CarNDTrackData2 master*
carnd-term1 ❯ cat ../CarNDTrackData/rightccw/driving_log.csv | wc -l
cat: ../CarNDTrackData/rightccw/driving_log.csv: No such file or directory
       0

~/projects/CarND-Behavioral-Cloning-P3/CarNDTrackData2 master*
carnd-term1 ❯ cat ../CarNDTrackData/rightcw/driving_log.csv | wc -l
    2803

~/projects/CarND-Behavioral-Cloning-P3/CarNDTrackData2 master*
carnd-term1 ❯ cat ../CarNDTrackData/leftcw/driving_log.csv | wc -l
    2277

~/projects/CarND-Behavioral-Cloning-P3/CarNDTrackData2 master*
carnd-term1 ❯ cat ../CarNDTrackData/recoveries/driving_log.csv | wc -l
     578

~/projects/CarND-Behavioral-Cloning-P3/CarNDTrackData2 master*

---
cat ../CarNDTrackData/recoveries/driving_log.csv > driving_log.csv
cat ../CarNDTrackData/leftccw/driving_log.csv >> driving_log.csv
cat ../CarNDTrackData/leftcw/driving_log.csv >> driving_log.csv

so now we have on bigger training set of our own
 
 carnd-term1 ❯ cat driving_log.csv | wc -l
    4852
    
    
