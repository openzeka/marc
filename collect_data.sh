#!/bin/bash
source devel/setup.bash
source ../cv_bridge_python3/install/setup.bash --extend 
rosrun deep_learning collect_data.py
