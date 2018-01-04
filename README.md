# Imitation_Learning_Quadcopter

I have done this project with DCU_Lab members!!! 

## Implementation

### [Setting]
#### Hardware
1. Flight Controller and Camera: RaspberryPi3 + Navio2 + PiCamera
2. Frame: dji s500
#### Software
1. Dronekit
<img src = "https://user-images.githubusercontent.com/34183439/34472363-f3fbd16c-efa3-11e7-9c55-a00501aefd2a.jpg" align="right" width="400" height="400">
<img src = "https://user-images.githubusercontent.com/34183439/34472364-f52b7150-efa3-11e7-9afc-973800f94b7d.jpg" alight="left" width="400" height="400" >


## How to extract control signal and images
1. After turning on raspberrypi3 + navio2, connect first (ID:pi, Password:raspberry)
2. Run pi_camera.py script(sudo python pi_camera.py)
3. Control to fly drone
4. You can see as below that control input(roll) and images are saved simultaneously.  
<img src = "https://user-images.githubusercontent.com/34183439/34472112-7b7e9f18-ef9d-11e7-8a4b-ab862e034afb.gif" width="600" height="400">


## How to do imitation learning

1. Run imitation_learning_training.py script to train
2. Run imitation_learning_test.py script to extract roll value(test)


