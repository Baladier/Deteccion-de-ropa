
# ROS Clothing Detection 

A package for the detection of clothings in images. This project is based on the [ailia models repository](https://github.com/axinc-ai/ailia-models/tree/master/deep_fashion/clothing-detection).

## Installation

1. Create and activate a [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment:
```
conda create -n clothing_det_env python=3.8
conda activate clothing_det_env
```

2. Install clothing-detector dependencies:
```
pip install ailia
git clone https://github.com/axinc-ai/ailia-models.git
cd ailia-models
pip install -r requirements.txt

```

3. Install ros-python dependencies:
```
pip install rospy rospkg pyyaml
```

4. Compile ROS package:
```
roscd;cd ..
catkin_make

```
T5. Pasos que realice:
```
mkdir -p ~/clothingtest/src
cd clothingtest/
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin_make
```

Pasos para correr los programas:
```
roscore (En otra terminal)
conda activate clothing_det_env
cd clothingtest
source /opt/ros/noetic/setup.bash
source devel/setup.bash
cd src/clothing_detection/src
python clothing_detector_node.py

En otra terminal..

cd ..
cd scripts
python3 camara_node.py
python visualization.py
...


```
