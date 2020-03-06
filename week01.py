import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# tensorboard logging
logdir = "logs/scalars/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch = 100000000)

# artficial data
batch = 100
H = 10*batch

A = np.array([[0, 1], [-.5, 1]])
b = np.array([1, 0])
c = np.array([0, 1])
x = np.zeros((2, H))

x[:, 0] = np.random.rand(2)
y = np.zeros(H)
u = np.zeros(H)
t = np.zeros(H)
for i in range(1, H):
    t[i] = i
    u[i] = 0.01*np.sin(2*np.pi/100*i)
    x[:, i] = np.matmul(A, x[:, i-1]) + b*u[i]
    y[i] = np.matmul(c, x[:, i])

plt.plot(u)
plt.plot(y)
plt.show()
#close figure to continue...

trainU = u.reshape((batch, int(H/batch), 1))
trainY = y.reshape((batch, int(H/batch), 1))

# simple recurrent neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(2))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainU, trainY, batch_size=batch, epochs=100, callbacks=[tensorboard_callback])

# start tensorboard using anaconda prompt
# > tensorboard --logdir logs\scalars
# see https://www.tensorflow.org/tensorboard/get_started