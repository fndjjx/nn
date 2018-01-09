import tensorflow as tf
import numpy as np
import pandas as pd
import better_exceptions


def transform_data(l):
    #input [[5, 7, 8], [6, 3], [3], [1]]
    #output [[5,6,3,1],[7,3,0,0],[8,0,0,0]] 
    max_time_step = max([len(i) for i in l])
    data = np.zeros([max_time_step, len(l), 1])
    for s in range(max_time_step):
        for b in range(len(l)):
            try:
                data[s][b] = [l[b][s]]
            except:
                pass
    return data

def generate_data(batch_size):
    data = []
    for i in range(batch_size):
        length = int(np.random.uniform(1,10))
        seq = []
        for j in range(length):
            seq.append(int(np.random.uniform(2,10)))
        data.append(seq)
    return data

def next_feed():
    batch = generate_data(10)
    encoder_input = transform_data(batch)
    decoder_target = transform_data([s+[EOS] for s in batch])
    decoder_input = transform_data([[EOS]+s for s in batch])
    return encoder_input, decoder_target, decoder_input

def generate_data2(data, batch_size):
    input_data = []
    output_data = []
    whole_data_length = len(data)
    for i in range(batch_size):
        start = int(np.random.uniform(0,whole_data_length-20))
        input_seq = data[start:start+10]
        output_seq = data[start+10:start+20]
        
        input_data.append(input_seq)
        output_data.append(output_seq)
    return input_data, output_data

def next_feed_file(file_name, column):
    data = list(pd.read_csv(file_name)[column].values)
    batch_input, batch_output = generate_data2(data, 100)
    encoder_input = transform_data(batch_input)
    decoder_target = transform_data([s+[1] for s in batch_output])
    decoder_input = transform_data([[1]+s for s in batch_output])
    return encoder_input, decoder_target, decoder_input
           
        


tf.reset_default_graph()

encoder_hidden_size = 20
decoder_hidden_size = 20

EOS = 1
PAD = 0

encoder_input = tf.placeholder(tf.float32, shape=[None, None, 1],name="encoder_input")
decoder_target = tf.placeholder(tf.float32, shape=[None, None, 1],name="decoder_target")
decoder_input = tf.placeholder(tf.float32, shape=[None, None, 1],name="decoder_input")


#encoder
encoder_cell = tf.contrib.rnn.GRUCell(num_units=encoder_hidden_size)
_, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input, dtype=tf.float32, time_major=True, scope="encoder")
print(encoder_final_state)

#decoder
decoder_cell = tf.contrib.rnn.GRUCell(num_units=decoder_hidden_size)
decoder_output, _ = tf.nn.dynamic_rnn(decoder_cell, decoder_input, initial_state=encoder_final_state, dtype=tf.float32, time_major=True, scope="decoder")
decoder_output = tf.layers.dense(decoder_output, 1)


#loss
loss = tf.reduce_sum(tf.pow(tf.reshape(decoder_output,[-1])-tf.reshape(decoder_target,[-1]),2))
train_op = tf.train.AdamOptimizer().minimize(loss)


def test1():
    e = [[5,7,8],[6,3],[3],[2]]
    e = transform_data(e)
    d = [[1,1,1,1],[1,1,1],[1,1],[1,1]]
    d = transform_data(d)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    pred = sess.run([decoder_prediction], feed_dict={encoder_input:e, decoder_input:d})
    print(e)
    print(pred)

def test2():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    max_batch = 300
    for i in range(max_batch):
        ei, dt, di = next_feed()
        feed = {
            encoder_input:ei,
            decoder_input:di,
            decoder_target:dt
        }
        result = sess.run([train_op, loss, decoder_prediction, decoder_output], feed_dict=feed)
        print(result[1])
        if i % 100 == 0:
            print(ei[:,0])
            print(result[2][:,0])
            print(result[3])

def test3():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    max_batch = 3000
    for i in range(max_batch):
        ei, dt, di = next_feed_file("data.csv","close")
        feed = {
            encoder_input:ei,
            decoder_input:di,
            decoder_target:dt
        }
        result = sess.run([train_op, loss, decoder_output], feed_dict=feed)
        print(result[1])
        if i%500==0:
            print(ei[:,0])
            print(dt[:,0])
            print(result[2][:,0])
         



test3()




        
