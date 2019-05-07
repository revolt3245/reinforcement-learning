import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.InteractiveSession()
Image = np.array([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]],dtype = np.float32)

plt.imshow(Image.reshape(3,3),cmap = 'Greys')