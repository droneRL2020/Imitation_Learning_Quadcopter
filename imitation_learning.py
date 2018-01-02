import numpy as np
import csv
from PIL import Image
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


action_list=[]
img_list=[]


##=== Load action(roll) input values
file = open('roll_record.csv', 'r', encoding='utf-8')
saved_file = csv.reader(file)

##=== Stack normalized action input
action_list = []
for line in saved_file:
    action = int(line[1])
    action = (action - 1496) / 2011 # Normalize control signal 981~2011 to -1~1
    action_list.append(action)
#print(action_list)
file.close()  


##=== Load images and reshape
for i in range(0,89):
    input_img = Image.open("D:\Imitation_Learning_Quadcopter\saved_image\image_%d.jpg" % i)
    img_list.append(input_img)
    
def img_reshape(input_img):
    """ (3, 64, 64) --> (64, 64, 3) """
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.resize(_img, (1, 64, 64, 3))
    return _img

##=== Packing data
num_actions = 1  # Get roll input only now
images_all = np.zeros((0, 64, 64, 3))  # initialize images_all
actions_all = np.zeros((0, num_actions)) # 0 = none
#rewards_all = np.zeros((0,)) # Later planning to do reinforcement learning
print("#"*50)  
print('Packing data into arrays... ')
for img, act in zip(img_list, action_list): # Have loaded img_list => reshape => stacking
    images_all = np.concatenate([images_all, img_reshape(img)], axis=0)  # packing at images_all
    actions_all = np.concatenate([actions_all, np.reshape(act, [1,num_actions])], axis=0) # packing action
    
#images_all.shape
#actions_all.shape 

##=== Save the expert's data
tl.files.save_any_to_npy(save_dict={'im': images_all, 'act': actions_all}, name='_tmp.npy')

# save every 10th expert's observation to train 나중에 10으로 바꿔
# can check in teacher's folder
tl.files.exists_or_mkdir('image/teacher', verbose=True)
for i in range(0, len(images_all), 1):
    tl.vis.save_image(images_all[i], 'image/teacher/im_%d.png' % i)
    
img_dim = [64, 64, 3]
n_action = 1        # steer only (float, left and right 1 ~ -1)
steps = 1000        # maximum step for a game
batch_size = 32
n_epoch = 100

###================= Define model
class Agent(object):
    def __init__(self, name='model', sess=None):
        assert sess != None
        self.name = name
        self.sess = sess

        self.x = tf.placeholder(tf.float32, [None, 64, 64, 3], name='Observaion')
        self.y = tf.placeholder(tf.float32, [None, 1], name='Steer')

        self._build_net(True, False)
        self._build_net(False, True)
        self._define_train_ops()

        tl.layers.initialize_global_variables(self.sess)

        print()
        self.n_test.print_layers()
        print()
        self.n_test.print_params(False)
        print()
        # exit()

    def _build_net(self, is_train=True, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)

            n = InputLayer(self.x / 255, name='in')

            n = Conv2d(n, 32, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c1/1')
            n = Conv2d(n, 32, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c1/2')
            n = MaxPool2d(n, (2, 2), (2, 2), 'VALID', name='max1')

            n = DropoutLayer(n, 0.75, is_fix=True, is_train=is_train, name='drop1')

            n = Conv2d(n, 64, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c2/1')
            n = Conv2d(n, 64, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c2/2')
            n = MaxPool2d(n, (2, 2), (2, 2), 'VALID', name='max2')
            # print(n.outputs)
            n = DropoutLayer(n, 0.75, is_fix=True, is_train=is_train, name='drop2')

            n = FlattenLayer(n, name='f')
            n = DenseLayer(n, 512, tf.nn.relu, name='dense1')
            n = DropoutLayer(n, 0.5, is_fix=True, is_train=is_train, name='drop3')
            n = DenseLayer(n, 1, tf.nn.tanh, name='o')

        if is_train:
            self.n_train = n
        else:
            self.n_test = n

    def _define_train_ops(self):
        self.cost = tl.cost.mean_squared_error(self.n_train.outputs, self.y, is_mean=False)
        self.train_params = tl.layers.get_variables_with_name(self.name, train_only=True, printable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost, var_list=self.train_params)

    def train(self, X, y, n_epoch=100, batch_size=10, print_freq=20):
        for epoch in range(n_epoch):
            start_time = time.time()
            total_err, n_iter = 0, 0
            for X_, y_ in tl.iterate.minibatches(X, y, batch_size, shuffle=True):
                _, err = self.sess.run([self.train_op, self.cost], feed_dict={self.x: X_, self.y: y_})
                total_err += err
                n_iter += 1
            if epoch % print_freq == 0:
                print("Epoch [%d/%d] cost:%f took:%fs" % (epoch, n_epoch, total_err/n_iter, time.time()-start_time))

    def predict(self, image):
        a = self.sess.run(self.n_test.outputs, {self.x : image})
        return a

    def save_model(self):
        tl.files.save_npz(self.n_test.all_params, name=self.name+'.npz', sess=self.sess)

    def load_model(self):
        tl.files.load_and_assign_npz(sess=self.sess, name=self.name+'.npz', network=self.n_test)
        
###===================== Pretrain model using data for demonstration
sess = tf.InteractiveSession()
model = Agent(name='model', sess=sess)
model.train(images_all, actions_all, n_epoch=n_epoch, batch_size=batch_size)
# save model after pretraining
model.save_model()
# model.load_model()
output_file = open('results.txt', 'w')
#for i in range(0, 89):                    ## When want to see and modify action values
#    output_file.write(str(actions_all[i])) 

class Roll:
    def __init__(self, test_image):
        self.test_image = test_image
        self.roll = model.predict(img_reshape(test_image)) *2011 + 1496
     
    def denormalize_action(self):
        return self.roll
