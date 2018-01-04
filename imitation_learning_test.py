from imitation_learning_training import Agent
from PIL import Image
import tensorflow as tf
import numpy as np


default_control_signal = 1496
max_control_signal = 2011

def img_reshape(input_img):
    """ (3, 64, 64) --> (64, 64, 3) """
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.resize(_img, (1, 64, 64, 3))
    return _img
###===== Gets test_image as input when drone flies
input_test_image = Image.open("D:\Imitation_Learning_Quadcopter\saved_image\image_%d.jpg" % 87)

model_1 = Agent(name='model',sess=tf.InteractiveSession())
model_1.load_model()

###===== Denormalize output(roll)
output_1 = model_1.predict(img_reshape(input_test_image)) * max_control_signal + default_control_signal
print(output_1)
