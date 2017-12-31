#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from dronekit import connect, VehicleMode
from time import sleep, time
import csv

print("\nConnecting to vehicle on: 127.0.0.1:14550")
vehicle = connect('127.0.0.1:14550', wait_ready=True)

vehicle.wait_ready('autopilot_version')

print(" Attitude: %s" % vehicle.attitude)
print(" Airspeed: %s" % vehicle.airspeed)  # settable
print(" Mode: %s" % vehicle.mode.name)  # settable
print(" Armed: %s" % vehicle.armed)  # settable

with open('roll_record.csv', 'wb') as csvfile:
    fieldname = ['time', 'roll_input']
    w = csv.DictWriter(csvfile, fieldnames=fieldname)
    w.writeheader()
    start = time()
    for i in range(100):
        now = time()
        sleep(0.2)
        print('roll input is : ', vehicle.channels['3'])
        w.writerow({'time': now-start, 'roll_input': vehicle.channels['3']})

print("Completed")

