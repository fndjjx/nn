import tensorflow as tf
import numpy as np
import pandas as pd


def transform_data(l):
    #input [[5, 7, 8], [6, 3], [3], [1]]
    #output [[5,6,3,1],[7,3,0,0],[8,0,0,0]] 
    max_time_step = max([len(i) for i in l])
    data = np.zeros([max_time_step, len(l)])
    for s in range(max_time_step):
        for b in range(len(l)):
            try:
                data[s][b] = l[b][s]
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
           
        


tf.reset_default_graph()

vocal_size = 10
embedding_size = 20
encoder_hidden_size = 20
decoder_hidden_size = 20

EOS = 1
PAD = 0

encoder_input = tf.placeholder(tf.int32, shape=[None, None],name="encoder_input")
decoder_target = tf.placeholder(tf.int32, shape=[None, None],name="decoder_target")
decoder_input = tf.placeholder(tf.int32, shape=[None, None],name="decoder_input")

#embdding
embeddings = tf.Variable(tf.random_uniform([vocal_size, embedding_size],-1,1))
encoder_input_embedding = tf.nn.embedding_lookup(embeddings, encoder_input)
decoder_input_embedding = tf.nn.embedding_lookup(embeddings, decoder_input)

#encoder
encoder_cell = tf.contrib.rnn.GRUCell(num_units=encoder_hidden_size)
_, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input_embedding, dtype=tf.float32, time_major=True, scope="encoder")
print(encoder_final_state)

#decoder
decoder_cell = tf.contrib.rnn.GRUCell(num_units=decoder_hidden_size)
decoder_output, _ = tf.nn.dynamic_rnn(decoder_cell, decoder_input_embedding, initial_state=encoder_final_state, dtype=tf.float32, time_major=True, scope="decoder")

#output
decoder_output = tf.layers.dense(decoder_output, vocal_size)
decoder_prediction = tf.argmax(decoder_output, 2)

#loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = decoder_output, labels=tf.one_hot(decoder_target, depth=vocal_size))
loss = tf.reduce_mean(cross_entropy)
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
         



test2()




        
