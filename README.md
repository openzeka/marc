# Open Zeka MARC
For vehicles with RPLidar

ZED Node will automatically initialized when you start the teleop.launch 
You do not need to start ZED node again.  

For collect data:
```bash
rosrun deep_learning collect_data.py
```

For driving autonomously:
```bash
rosrun deep_learning predict.py
```

Note that in order to run these nodes, python3 is required. You need to rebuild cv_bridge package for python3. 

Follow these steps: [cv_bridge for python3](https://github.com/openzeka/cv_bridge_python3)

Tested on Jetson TX2 with Jetpack 4.2 
