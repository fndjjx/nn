import tensorflow as tf
import numpy as np
from emd import one_dimension_emd
import pandas as pd
import better_exceptions
from sklearn.metrics import mean_squared_error, r2_score


class RNN():
    def __init__(self, n_input, n_step_input, n_step_output, n_hidden, n_output, n_layer):
        tf.reset_default_graph()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layer = n_layer

        self.x = tf.placeholder(tf.float32, [None, n_step_input, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_step_output, n_output])

        #cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_hidden, activation=tf.nn.relu), output_size = n_output)
        #cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden), output_size = n_output)
#        cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(num_units=n_hidden), output_size = n_output)
#        drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)

        #https://stackoverflow.com/questions/44615147/valueerror-trying-to-share-variable-rnn-multi-rnn-cell-cell-0-basic-lstm-cell-k
        def drop_cell():
            cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(num_units=n_hidden), output_size = n_output)
            drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)
            return drop_cell
        stacked_cell = [drop_cell() for _ in range(n_layer)]
        #drop_multi_layer_cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * n_layer)
        drop_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(stacked_cell)

        drop_rnn_output, states = tf.nn.dynamic_rnn(drop_multi_layer_cell, self.x, dtype=tf.float32)
        weight = self.weight_variable([n_step_input, n_step_output])
        bias = self.bias_variable([n_step_output])
        drop_rnn_output = tf.reshape(drop_rnn_output, [-1,n_step_input])
        self.output = tf.matmul(drop_rnn_output, weight)+bias

       
        
       
        #self.loss = tf.reduce_sum(tf.pow(tf.reshape(self.output,[-1])-tf.reshape(self.y,[-1]),2))
        self.loss = tf.reduce_sum(abs(tf.reshape(self.output,[-1])-tf.reshape(self.y,[-1]))/((abs(tf.reshape(self.output,[-1]))+abs(tf.reshape(self.y,[-1])))/2))
        optimizer = tf.train.AdamOptimizer()
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

    data = np.sin(np.array(range(101))*0.01)        
    x = []
    y = []
    for i in range(len(data)-21):
        x.append([[j] for j in data[i:i+20]])
        y.append([[data[i+20]]])
    print(x[0])
    print(y[0])
    print(x[1])
    print(y[1])
    rnn = RNN(1,20,1,500,1,2)
    for i in range(5000):
        #batch_index = np.random.randint(0,len(x))
        #xx = [x[batch_index]]
        #yy = [y[batch_index]]

        _,loss=rnn.fit(x,y)
        print(loss)

    output = rnn.predict([x[-1]])
    print(output)
    print(y[-1])
    x = np.sin(np.array(range(2001))*0.01)        
    output = rnn.predict([[[i] for i in x[-21:-1]]])
    print(output)
    print(x[-1])



def test2():
    import pandas as pd
    raw_data = pd.read_csv("/tmp/111.csv")["2NE1_zh.wikipedia.org_all-access_spider"].values
    data = raw_data[-540:-60]

    x = []
    y = []
    for i in range(len(data)-360):
        x.append([[j] for j in data[i:i+300]])
        y.append([[j] for j in data[i+300:i+360]])
    print(x[0])
    print(y[0])
    rnn = RNN(1,300,60,1000,1,5)
    for i in range(3000):
        #batch_index = np.random.randint(0,len(x))
        #xx = [x[batch_index]]
        #yy = [y[batch_index]]
        _,loss=rnn.fit(x,y)
        print(loss)

    x_test = [[[j] for j in data[-300:]]]
    y_test = raw_data[-60:]
    print(x_test)
    output = rnn.predict(x_test)
    output = np.ravel(output)
    y_test = np.ravel(y_test)
    print(output)

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

def test4():
    df = pd.read_csv("data.csv")
    data = df.values
    target = df.close.values

    x_train = data[:-100]
    y_train = target[:-100]


    x_test = data[-100:]
    y_test = target[-100:]

    x = []
    y = []
    p = 20
    for i in range(len(x_train)-p):
        x.append([j for j in x_train[i:i+p]])
        y.append([[y_train[i+p]]])
    print(x[1])
    print(y[0])

    x_test1=[]
    y_test1=[]
    for i in range(len(x_test)-p):
        x_test1.append([j for j in x_test[i:i+p]])
        y_test1.append([[y_test[i+p]]])
    print(x[-1])
    print(y[-2])
    rnn = RNN(6,p,1,1000,1,3)
    for i in range(5000):
    #    #batch_index = np.random.randint(0,len(x))
    #    #xx = [x[batch_index]]
    #    #yy = [y[batch_index]]

        _,loss=rnn.fit(x,y)
        print(loss)

    print("predict")
    pred =[]
    true = []
    for i in range(len(x_test1)):
        output = rnn.predict([x_test1[i]])
        print(output)
        print(y_test1[i])
        pred.append(output[0][0])
        true.append(y_test1[i][0][0])
    print(float(r2_score(true, pred)))
    #x = np.sin(np.array(range(2001))*0.01)
    #output = rnn.predict([[[i] for i in x[-21:-1]]])
    #print(output)
    #print(x[-1])

    

    

if __name__=="__main__":
    test4()
