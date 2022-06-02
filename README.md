# Personal trainer with computer vision



![curlbicep](images/curl.gif)

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
Videos and images for this project can be downloaded [here](https://pexels.com)

Also for video, webcam can be used.

## Personal trainer
For now, there is four scripts:

 - curl_bicep_video.py:
 - pose_module.py:
 - quickly_pose_image.py:
 - tools.py:

Ubicarse en la repository root and:

## Tests
Ubicarse en la repository root and:

```
pytest -xs tests/test_personal_trainer.py
```

For running coverage:

```
coverage run --rcfile=coverage.cfg -m pytest tests/
coverage report
```

