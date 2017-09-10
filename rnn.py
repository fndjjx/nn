import tensorflow as tf
import numpy as np
from emd import one_dimension_emd


class RNN():
    def __init__(self, n_input, n_step, n_hidden, n_output, n_layer):
        tf.reset_default_graph()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layer = n_layer

        self.x = tf.placeholder(tf.float32, [None, n_step, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_step, n_output])

        #cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_hidden, activation=tf.nn.relu), output_size = n_output)
        cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden), output_size = n_output)
        #cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(num_units=n_hidden), output_size = n_output)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layer)
        self.output, states = tf.nn.dynamic_rnn(multi_layer_cell, self.x, dtype=tf.float32)
       
        #self.loss = tf.reduce_sum(tf.pow(self.output-self.y,2))
        self.loss = tf.reduce_sum(abs(tf.reshape(self.output,[-1])-tf.reshape(self.y,[-1]))/((abs(tf.reshape(self.output,[-1]))+abs(tf.reshape(self.y,[-1])))/2))
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train = optimizer.minimize(self.loss)
 
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def weight_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape))

    def bias_variable(self, shape):
        return tf.Variable(tf.zeros(shape))


    def fit(self, x, y):
        return self.sess.run([self.train, self.loss],feed_dict={self.x:x,self.y:y})

    def predict(self, x):
        return self.sess.run((self.output),feed_dict={self.x:x})

    def close(self):
        self.sess.close()


def test():

    data = np.sin(range(101))        
    x_data = data[:-1]
    y_data = data[1:]
    x = np.reshape(x_data, [-1,20,1])
    y = np.reshape(y_data, [-1,20,1])
    print(x[-1])
    print(y[-1])
    rnn = RNN(1,20,100,1,2)
    for i in range(1000):
        _,loss=rnn.fit(x,y)
        print(loss)

    test_data = np.sin(np.array(range(21))+100)        
    x_data = test_data[:-1]
    x_test = np.reshape(x_data, [-1,20,1])
    output = rnn.predict(x_test)
    print(output)
    print(test_data[1:])


def test2():
    import pandas as pd
    raw_data = pd.read_csv("/tmp/111.csv")["2NE1_zh.wikipedia.org_all-access_spider"].values
    data = raw_data[-540:-60]

    x_data = data[:-60]
    y_data = data[60:]
    x = np.reshape(x_data, [-1,60,1])
    y = np.reshape(y_data, [-1,60,1])
    rnn = RNN(1,60,300,1,20)
    for i in range(100):
        batch_index = np.random.randint(0,len(x))
        xx = [x[batch_index]]
        yy = [y[batch_index]]
        _,loss=rnn.fit(xx,yy)
        print(loss)

    x_test = data[-60:]
    x_test = np.reshape(x_test, [-1,60,1])
    y_test = raw_data[-60:]
    output = rnn.predict(x_test)
    output = np.ravel(output)
    y_test = np.ravel(y_test)

    def smape(target,pred):
        up = abs(target-pred)
        down = (abs(target)+abs(pred))/2
        #down = (target+pred)/2
        return sum(up/down)/len(target)
    print(smape(y_test,np.array([np.median(raw_data[-49:])]*60)))
    print(smape(y_test,output))

def test3():
    def smape(target,pred):
        up = abs(target-pred)
        down = (abs(target)+abs(pred))/2
        #down = (target+pred)/2
        return sum(up/down)/len(target)
    import pandas as pd
    raw_data1 = pd.read_csv("/tmp/111.csv")["2NE1_zh.wikipedia.org_all-access_spider"].values
    #data = raw_data[-540:-60]
    myemd=one_dimension_emd(raw_data1)
    imfs,residual = myemd.emd(0.01,0.01)
    imfs.append(residual)

    outputs=[]
    #for index in range(len(imfs)):
    for index in range(1):
        print("imf{}".format(index))
        raw_data = imfs[index]
        data = raw_data[-540:-60]
        x_data = data[:-60]
        y_data = data[60:]
        x = np.reshape(x_data, [-1,60,1])
        y = np.reshape(y_data, [-1,60,1])
        rnn = RNN(1,60,300,1,1)
        for i in range(5000):
            #batch_index = np.random.randint(0,len(x))
            #xx = [x[batch_index]]
            #yy = [y[batch_index]]
            _,loss=rnn.fit(x,y)
            if i%100==0:
                print(i)

        x_test = data[-60:]
        x_test = np.reshape(x_test, [-1,60,1])
        y_test = raw_data[-60:]
        output = rnn.predict(x_test)
        output = np.ravel(output)
        y_test = np.ravel(y_test)
        print(smape(y_test,np.array([np.median(raw_data[-49:])]*60)))
        print(smape(y_test,output))
        print(y_test)
        print(output)
        outputs.append(output)
        rnn.close()

    print(outputs)
    output = sum(outputs)
    print(output)
    y_test = raw_data1[-60:]
    print(smape(y_test,np.array([np.median(raw_data1[-49:])]*60)))
    print(smape(y_test,output))
    

if __name__=="__main__":
    test3()
