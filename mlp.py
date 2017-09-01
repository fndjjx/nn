import tensorflow as tf
import numpy as np

def xavier_init(n_in, n_out):
    low = -np.sqrt(6/(n_in+n_out))
    high = np.sqrt(6/(n_in+n_out))
    return tf.random_uniform((n_in,n_out), minval=low, maxval=high, dtype=tf.float32)


class MLP():
    def __init__(self, n_input, n_hidden_list, n_output, activate_function, keep_prob=0.75):
        self.n_input = n_input
        self.n_hidden = n_hidden_list
        self.n_output = n_output
        self.weights = self.init_weights()
        self.optimizer = tf.train.AdamOptimizer()
        self.activate_function = activate_function
        self.keep_prob = keep_prob

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_output])
        self.hidden = []
        for i in range(len(self.n_hidden)):
            if i==0:
                hidden = tf.nn.dropout(self.activate_function(tf.add(tf.matmul(self.x, self.weights["w{}".format(i+1)]), self.weights["b{}".format(i+1)])),self.keep_prob)
            else:
                hidden = tf.nn.dropout(self.activate_function(tf.add(tf.matmul(self.hidden[i-1], self.weights["w{}".format(i+1)]), self.weights["b{}".format(i+1)])),self.keep_prob)
            self.hidden.append(hidden)
        w_output = i+2 
        self.output = tf.nn.softmax(tf.add(tf.matmul(self.hidden[-1], self.weights["w{}".format(w_output)]), self.weights["b{}".format(w_output)]))

        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.output, self.y),2))
#        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.output),reduction_indices=[1]))
        self.train = self.optimizer.minimize(self.cost)

        self.correct = tf.equal(tf.argmax(self.output,1),tf.argmax(self.y,1))
        self.acc = tf.reduce_mean(tf.cast(self.correct,tf.float32))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def init_weights(self):
        weights = {}
        for i in range(len(self.n_hidden)):
            if i==0:
                weights["w{}".format(i+1)] = tf.Variable(xavier_init(self.n_input, self.n_hidden[i]))
                weights["b{}".format(i+1)] = tf.Variable(tf.zeros([self.n_hidden[i]]))
            else:
                weights["w{}".format(i+1)] = tf.Variable(xavier_init(self.n_hidden[i-1], self.n_hidden[i]))
                weights["b{}".format(i+1)] = tf.Variable(tf.zeros([self.n_hidden[i]]))
        w_output = i+2 
        weights["w{}".format(w_output)] = tf.Variable(xavier_init(self.n_hidden[-1], self.n_output))
        weights["b{}".format(w_output)] = tf.Variable(tf.zeros([self.n_output]))
        return weights

    def fit(self, x, y):
        self.sess.run((self.cost, self.train),feed_dict={self.x:x, self.y:y})

    def predict(self, x, y=None):
        if y:
            return self.sess.run((self.output, self.acc),feed_dict={self.x:x, self.y:y})
        else:
            return self.sess.run((self.output),feed_dict={self.x:x})

    def validation_cost(self, x, y):
        return self.sess.run((self.cost),feed_dict={self.x:x, self.y:y})




def test():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    print(x_train.shape)
    print(y_train.shape)

    x_test = mnist.test.images
    y_test = mnist.test.labels

    mlp = MLP(784,[500,500],10,tf.nn.relu)
    for i in range(100):
        print(i)
        x_batch, y_batch = mnist.train.next_batch(100)
        mlp.fit(x_train, y_train)

    print(mlp.validation_cost(x_test, y_test))
    print(mlp.predict(x_test, y_test))

def test2():
    from sklearn.datasets import make_regression
    from sklearn.cross_validation import train_test_split
    x,y = make_regression()
    y = y[:,np.newaxis]
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    mlp = MLP(100,[10],1,tf.nn.relu)
    for i in range(20):
        mlp.fit(x_train, y_train)

    print(mlp.validation_cost(x_test, y_test))

    mlp = MLP(100,[3000, 3000, 3000, 3000],1,tf.nn.relu)
    for i in range(20):
        mlp.fit(x_train, y_train)

    print(mlp.validation_cost(x_test, y_test))

def test3():
    from datacleaner import autoclean
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    train_df = pd.read_csv("../dataset/titanic/train_tita.csv")
    test_df = pd.read_csv("../dataset/titanic/test_tita.csv")
    train_df = autoclean(train_df)
    test_df = autoclean(test_df)
    target = "Survived"

    #df = train_df
    #y = df[target].values
    #x = df.drop(target,axis=1).values
    #y = [[0,1]if i==1 else [1,0] for i in y]
    #x_train, x_test, y_train, y_test = train_test_split(x,y)
    #mlp = MLP(11,[50,30,30],2,tf.nn.relu)
    #for i in range(100000):
    #    mlp.fit(x_train, y_train)

    #print(mlp.validation_cost(x_test, y_test))
    #print(mlp.predict(x_test, y_test))

    y = train_df[target].values
    x = train_df.drop(target,axis=1).values
    x_test = test_df.values
    y = [[0,1]if i==1 else [1,0] for i in y]
    mlp = MLP(11,[50,30,30],2,tf.nn.relu)
    for i in range(100000):
        mlp.fit(x, y)
    result = mlp.predict(x_test)
    result = np.array([0 if i[0]>i[1] else 1 for i in result])
    print(result)
   
    id = test_df["PassengerId"]
    result = pd.DataFrame({"PassengerId":id,"Survived":result})
    result.to_csv("subminssion.csv",index=False)



if __name__=="__main__":
    test3()
