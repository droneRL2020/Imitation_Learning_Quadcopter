from imitation_learning import Roll
from PIL import Image


###===== Gets test_image as input when drone flies
input_test_image = Image.open("D:\Imitation_Learning_Quadcopter\saved_image\image_%d.jpg" % 87)
expert_1 = Roll(input_test_image)

output_action = Roll.denormalize_action(expert_1)
print(output_action)