
# Personal trainer using computer vision
<img src="images/curl.jpg" width=70%>

<img src="./images/curl.jpg" width=70% align="center"> 

Module to count the correct repetitions of bicep curls using mediapipe and opencv.

An example:

![curlgif](images/curl.gif)


## Setup environment
```
conda create --name pt-env-py310 python=3.10
pip install -r requirements.txt
```

For performing tests:
```
pip install pytest, coverage
```

## Data
The videos and images for this project can be downloaded [here](https://pexels.com)


## Personal trainer
For now, there are four scripts:

 - curl_bicep_video.py: module of dumbbell biceps curl to count the correct repetitions.
 - pose_module.py: pose estimation for the detection of different types of exercises.
 - quickly_pose_image.py: run and get quickly the landmarks in an image.
 - tools.py: useful tools for the computer vision projects.


## Tests
Go to the repository root and run this:

```
pytest -xs tests/test_personal_trainer.py
```

For running the coverage:

```
coverage run --rcfile=coverage.cfg -m pytest tests/
coverage report
```
The project coverage is 0.98.

