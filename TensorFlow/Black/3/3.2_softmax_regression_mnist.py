import tensorflow as tf
import matplotlib.pyplot as plt

# read data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalize data
x_train, x_test = x_train / 255, x_test / 255

# one hot encoding
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

# show first image
# plt.imshow(x_train[0])
# plt.colorbar()
# plt.show()


class Softmax(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super(Softmax, self).__init__()
        self.w = tf.Variable(tf.zeros([input_shape, 10]), trainable=True)
        self.b = tf.Variable(tf.zeros([10]), trainable=True)

    def call(self, inputs, **kwargs):
        return tf.nn.softmax(tf.matmul(inputs, self.w) + self.b)


# create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(dtype=tf.float32, input_shape=(x_train.shape[1], x_train.shape[2])),
    Softmax(input_shape=x_train.shape[1] * x_train.shape[2])
])

def cross_entropy(y_true, y_pred):
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


optimizer = tf.keras.optimizers.SGD(0.5)
model.compile(loss=cross_entropy, optimizer=optimizer,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

model.evaluate(x_test,  y_test, verbose=2)

