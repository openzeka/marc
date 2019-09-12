#!/bin/bash

IMU=/dev/ttyACM1
VESC=/dev/ttyACM0
LIDAR=/dev/ttyUSB0
if [ "$1" != "no-check" ]; then
	if [ ! -c "$IMU" ] || [ ! -c "$VESC" ]; then
		echo "Device not found: IMU or VESC not found -> $IMU $VESC"
		exit
	else 
		sudo chmod 777 $IMU
		sudo chmod 777 $VESC
	fi

	if [ ! -c "$LIDAR" ]; then
		echo "Device not found: LIDAR -> $LIDAR"
		exit
	else
		sudo chmod 777 $LIDAR
	fi
else
	echo "Not checking..."
	sudo chmod 777 $IMU
	sudo chmod 777 $VESC
	sudo chmod 777 $LIDAR
fi
source devel/setup.bash
roslaunch racecar teleop.launch

	
