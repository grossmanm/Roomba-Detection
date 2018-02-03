from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import listdir
from os.path import isfile, join
import tensorflow as tf
import cv2
import numpy as np
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
print(var.shape)
print(var.dtype)
print(var.eval)
x = tf.constant([[1],[2],[3],[4]],dtype=tf.float32)
y_true = tf.constant([[0],[-1],[-2],[-3]], dtype=tf.float32)
tensor = x * y_true
t = tf.Print(tensor, [tensor])
result = t +1
print(result)
linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)



write = tf.summary.FileWriter('abcd')
var_init = tf.global_variables_initializer()
loss = tf.losses.mean_squared_error(labels=y_true,predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    _, loss_value = sess.run((train,loss))

print(sess.run(y_pred))
write.add_graph(tf.get_default_graph())
