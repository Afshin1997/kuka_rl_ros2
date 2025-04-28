
# Camera Settings

## Capturing Chessboard Images

To calibrate the camera, you need to capture images of a chessboard pattern. The following Python script will automatically capture images every 10 seconds. Ensure that the chessboard is always within the camera frame and move it slightly every 10 seconds to capture different perspectives. The images will be saved in the **calibration_images** directory.


### Steps
-1 Navigate to the directory:

```sh
cd ~/kuka_repo_ros/D435/
```

-2 Run the chessboard capture script:
```sh
python3 calibration/capture_chessboard.py 
```
## Intrinsic Calibrating the Camera

After capturing the images, generate the camera matrix and distortion coefficients by running the calibration script. Before executing the script, update the **chessboard_size** variable in the **intrinsic_calibration_D435I.py** script to match the actual size of your chessboard.


### Steps
1- Run the intrinsic calibration script:
```sh
python3 calibration/intrinsic_calibration_D35I.py
```
The generated calibration files will be saved in the **camera_params** directory, and will be used in ball tracking project.

## Extrinsic Calibrating the Camera

### Steps

1- Run the extrinsic calibration script:

```sh
python3 calibration/extrinsic_calibration_D35I.py
```

While this script is running, the chessboard should be positioned at the desired location. During this time, you can change the position of the chessboard as much as you want, but ensure that the coordinate system is visible in the captured image. The recorded extrinsic parameters will correspond to the last frame in which the camera detects the chessboard.

2- Run the following script in the case of having an additional rotation and translation of the coordinate systems after generation of chessboard coordinates. For example, if you want to move the center of the chessboard to the center of the baselink of the manipulaator.

```sh
python3 calibration/extrinsic_adjustment.py
```
this script saves the **camera_pose_base** file as the adjusted paprameters to the new coordinate system