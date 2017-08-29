import tensorflow as tf
import numpy as np

def xavier_init(n_in, n_out):
    low = -np.sqrt(6/(n_in+n_out))
    high = np.sqrt(6/(n_in+n_out))
    return tf.random_uniform((n_in,n_out), minval=low, maxval=high, dtype=tf.float32)

class AutoEncoder():
    def __init__(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.activation_function = tf.nn.softplus
        self.weights = self.init_weights()
        self.optimizer = tf.train.AdamOptimizer()

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.activation_function(tf.add(tf.matmul(self.x, self.weights["w1"]), self.weights["b1"]))
        self.output = tf.add(tf.matmul(self.hidden, self.weights["w2"]), self.weights["b2"])

        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.output, self.x), 2))
        self.train_step = self.optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def init_weights(self):
        weights = {}
        weights["w1"] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        weights["w2"] = tf.Variable(xavier_init(self.n_hidden, self.n_input))
        weights["b1"] = tf.Variable(tf.zeros([self.n_hidden]))
        weights["b2"] = tf.Variable(tf.zeros([self.n_input]))
        return weights

    def fit(self, x):
        cost , train = self.sess.run((self.cost, self.train_step), feed_dict={self.x: x})
        return cost

    def total_cost(self, x):
        cost  = self.sess.run(self.cost, feed_dict={self.x: x})
        return cost

    def transform(self, x):
        return self.sess.run(self.hidden, feed_dict={self.x:x})

 
def test():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train = mnist.train.images
    x_test = mnist.test.images
    print(x_train.shape)

    ae = AutoEncoder(784,200)
    for i in range(20):
        ae.fit(x_train)

    print(ae.total_cost(x_test))

def test2():
    import pandas as pd
    train_df = pd.read_csv("/tmp/train_1.csv")
    x_train = np.array([train_df.loc[0].values[1:], train_df.loc[1].values[1:]])
    x_test = [train_df.loc[2].values[1:]]
    print(x_train.shape)
    ae = AutoEncoder(len(x_train[0]),50)
    for i in range(20):
        ae.fit(x_train)
    print(ae.total_cost(x_test))
    print(ae.transform(x_test))
    print(x_test)

        
if __name__ == "__main__":
    test2()
