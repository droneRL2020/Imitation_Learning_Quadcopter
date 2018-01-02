from Imitation_Learning_Quad import Roll
from PIL import Image


###===== Gets test_image as input when drone flies
#test_image = Image.open("D:\Imitation_Learning_Quadcopter\saved_image\image_%d.jpg" % 87)
action = Roll()
roll_output = action.denormalize_action(test_image)