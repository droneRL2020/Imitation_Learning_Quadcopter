from __future__ import print_function
import time
import picamera
from dronekit import connect, VehicleMode
from time import sleep, time
import csv

print("\nConnecting to vehicle on: 127.0.0.1:14550")
vehicle = connect('127.0.0.1:14550', wait_ready=True)
vehicle.wait_ready('autopilot_version')

with open('roll_record.csv', 'wb') as csvfile:
    with picamera.PiCamera() as camera:
        camera.resolution = (120, 80)
        fieldname = ['time', 'roll_input', 'image_name']
        w = csv.DictWriter(csvfile, fieldnames=fieldname)
        w.writeheader()
        start = time()

        camera.start_preview()
        print('picamera warm up')
        sleep(2)

        for i in range(1000):
            now = time()

            name = 'save_image/image_' + str(i) + '.jpg'
            my_file = open(name, 'wb')

            camera.capture(my_file)
            print('roll input is : ', vehicle.channels['1'])
            print('save image : ', name)
            w.writerow({'time': now-start, 'roll_input': vehicle.channels['3'],
                        'image_name': name})
            # w.writerow({'time': now - start, 'roll_input': vehicle.channels['1']})
            my_file.close()

print("Completed")
# Explicitly open a new file called my_image.jpg
