from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import listdir
from os.path import isfile, join
import tensorflow as tf
import cv2
import numpy as np
# getting started with creating tensor objects from images and using arbitary values
path = 'C:/Users/Malcolm/Documents/GitHub/Roomba-Detection/roomba_photos/roomba/$_1.jpg'
out = []
im = cv2.imread(path)
im = cv2.resize(im, (32, 32), cv2.INTER_CUBIC)
im = (np.array(im))
r = im[:, :, 0].flatten()
g = im[:, :, 1].flatten()
b = im[:, :, 2].flatten()
label = [1]
out =np.array(list(label)+ list(r) + list(g) + list(b), np.uint8)
slices = tf.data.Dataset.from_tensor_slices(out)
next_item = slices.make_one_shot_iterator().get_next()
var = tf.Variable(out, tf.int16)
#print(var.shape)
#print(var.dtype)
#print(var.eval)
x = tf.constant([[1],[2],[3],[4]],dtype=tf.float32)
y_true = tf.constant([[0],[-1],[-2],[-3]], dtype=tf.float32)
tensor = x * y_true
t = tf.Print(tensor, [tensor])
result = t +1
#print(result)
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

# variable tutorial
# variables are tensors that can be changed, these changes
# can be seen across sessions
my_variable = tf.get_variable("my_variable", [1,2,3], dtype=tf.int32,
                              initializer=tf.zeros_initializer)

other_variable = tf.get_variable("other_variable", dtype=tf.int32,
                                 initializer=tf.constant([23,42]))

my_local = tf.get_variable("my_local", shape=(),
                          collections=[tf.GraphKeys.LOCAL_VARIABLES])

my_non_trainable = tf.get_variable("my_non_trainable",
                                   shape=(),
                                   trainable=False)
tf.add_to_collection("my_collection_name",my_local)
tf.get_collection("my_collection_name")

v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
with tf.control_dependencies([assignment]):
    w = v.read_value()


# GRAPHS AND SESSIONS TUTORIAL
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer
loss = ...
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
    writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

    for i in range(1000):
        sess.run(train_op)

    writer.close()

    y_val, output_val = sess.run([y,output])
""""
# execution for a session
write = tf.summary.FileWriter('abcd')
var_init = tf.global_variables_initializer()
loss = tf.losses.mean_squared_error(labels=y_true,predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess.run(my_variable.initializer)
print(sess.run(tf.report_uninitialized_variables()))

sess.run(assignment)

for i in range(100):
    _, loss_value = sess.run((train,loss))

print(sess.run(y_pred))
write.add_graph(tf.get_default_graph())
"""