import tensorflow as tf
import numpy as np


class CNN():
    def __init__(self, input_length, input_size, output_length, conv_size, pool_size, fc_size, dropout):
        self.input_size = input_size
        self.output_length = output_length
        self.conv_size = conv_size
        self.pool_size = pool_size
        self.activate_function = tf.nn.relu
        self.dropout = dropout
        self.fc_size = fc_size
        self.weights = self.init_weights()
        self.optimizer = tf.train.AdamOptimizer()

        self.x = tf.placeholder(tf.float32, [None, input_length])
        self.xx = tf.reshape(self.x, input_size)
        self.y = tf.placeholder(tf.float32, [None, output_length])

        self.conv_layer = []
        for i in range(len(conv_size)):
            if i == 0:
                conv = self.conv2d(self.xx, self.weights["conv_w_{}".format(i+1)], self.weights["conv_b_{}".format(i+1)], self.conv_size[i][1])
            else:
                conv = self.conv2d(self.conv_layer[i-1], self.weights["conv_w_{}".format(i+1)], self.weights["conv_b_{}".format(i+1)], self.conv_size[i][1])
            act = self.activate_function(conv)
            pool = self.pool(act, self.pool_size[i][0], self.pool_size[i][1])
            self.conv_layer.append(pool)

        self.fc_layer = []
        fc_input = tf.reshape(self.conv_layer[-1],[-1,7*7*8])
        
        for i in range(len(self.fc_size)-1):
            print(i)
            print(fc_input.dtype)
            fc = self.activate_function(tf.matmul(fc_input, self.weights["fc_w_{}".format(i+1)])+self.weights["fc_b_{}".format(i+1)])
            dropout = tf.nn.dropout(fc, self.dropout)
            fc_input = dropout
            self.fc_layer.append(dropout)

        w_output = i+2
        self.output = tf.nn.softmax(tf.matmul(self.fc_layer[-1],self.weights["fc_w_{}".format(w_output)])+self.weights["fc_b_{}".format(w_output)])
        self.loss = tf.reduce_sum(tf.pow(tf.subtract(self.output, self.y),2))
        self.train = self.optimizer.minimize(self.loss)

        self.correct = tf.equal(tf.argmax(self.output,1),tf.argmax(self.y,1))
        self.acc = tf.reduce_mean(tf.cast(self.correct,tf.float32))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def weight_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape))

    def bias_variable(self, shape):
        return tf.Variable(tf.zeros(shape))

    def conv2d(self, x, w, b, step):
        return tf.nn.conv2d(x, w, strides=step, padding="SAME") + b

    def pool(self, x, size, step):
        return tf.nn.max_pool(x, size, strides=step, padding="SAME")

    def init_weights(self):
        weights = {}
        for i in range(len(self.conv_size)):
            weights["conv_w_{}".format(i+1)] = self.weight_variable(self.conv_size[i][0])
            weights["conv_b_{}".format(i+1)] = self.bias_variable([self.conv_size[i][0][-1]])

        for i in range(len(self.fc_size)):
            if i==0:
                weights["fc_w_{}".format(i+1)] = self.weight_variable([7*7*8, self.fc_size[i]])
                weights["fc_b_{}".format(i+1)] = self.bias_variable([self.fc_size[i]])
            else:
                weights["fc_w_{}".format(i+1)] = self.weight_variable([self.fc_size[i-1], self.fc_size[i]])
                weights["fc_b_{}".format(i+1)] = self.bias_variable([self.fc_size[i]])
        
        return weights

    def fit(self, x, y):
        self.sess.run((self.loss, self.train),feed_dict={self.x:x, self.y:y})

    def predict(self, x, y=None):
        return self.sess.run((self.output, self.acc),feed_dict={self.x:x, self.y:y})

    def validation_loss(self, x, y):
        return self.sess.run((self.loss),feed_dict={self.x:x, self.y:y})
        
      
def test():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    print(x_train.shape)
    print(y_train.shape)

    x_test = mnist.test.images
    y_test = mnist.test.labels

    cnn = CNN(784, [-1,28,28,1], 10, [[[5,5,1,4],[1,1,1,1]], [[5,5,4,8],[1,1,1,1]]], [[[1,2,2,1],[1,2,2,1]],[[1,2,2,1],[1,2,2,1]]], [1000,10], 0.7)
    for i in range(2000):
        print(i)
        x_batch, y_batch = mnist.train.next_batch(10)
        cnn.fit(x_train, y_train)

    print(cnn.validation_loss(x_test, y_test))
    print(cnn.predict(x_test, y_test))

if __name__=="__main__":
    test()
        
