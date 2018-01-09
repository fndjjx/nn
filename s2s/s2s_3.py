import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.layers.core import Dense
import better_exceptions

def _get_simple_lstm(rnn_size, layer_size):
    lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
    return tf.contrib.rnn.MultiRNNCell(lstm_layers)

with open("source") as f:
    source_data1 = f.readlines()
    source_data_original = []
    for i in source_data1:
        source_data_original.append(i[:-1])
    source_data_original = [list(i) for i in source_data_original]
    
with open("target") as f:
    target_data1 = f.readlines()
    target_data_original = []
    for i in target_data1:
        target_data_original.append(i[:-1])
    target_data_original = [list(i) for i in target_data_original]

def generate_dictionary(data):
    char_data = [j for i in data for j in i]
    special_words = ["<EOS>", "<PAD>", "<GO>"]
    index2vocab = {index: value for index, value in enumerate(set(char_data + special_words))}
    vocab2index = {vocab: index for index, vocab in index2vocab.items()}
    return index2vocab, vocab2index

def fill_data(data):
    max_length = max([len(i) for i in data])
    for i in data:
        if len(i)<max_length:
            for j in range(max_length-len(i)):
                i.append("<PAD>")
    return data

def map_index2word(l, dictionay):
    s = ""
    for i in l:
        s+=dictionay[i]
    return s

source_index2vocab, source_vocab2index = generate_dictionary(source_data_original)
target_index2vocab, target_vocab2index = generate_dictionary(target_data_original)


target_data = fill_data(target_data_original)
source_data = fill_data(source_data_original)
max_length = max([len(i) for i in target_data_original])
source_transform = [[source_vocab2index[j] for j in i] for i in source_data]
target_transform = [[target_vocab2index[j] for j in i] for i in target_data]

def get_encoder(encoder_input, embedding_size, vocab_size, encoder_hidden_size):
    #embdding
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1,1))
    encoder_input_embedding = tf.nn.embedding_lookup(embeddings, encoder_input)
    
    #encoder
    #encoder_cell = tf.contrib.rnn.GRUCell(num_units=encoder_hidden_size)
    encoder_cell =  tf.contrib.rnn.MultiRNNCell([get_encoder_cell(encoder_hidden_size) for i in range(3)])
    encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input_embedding, dtype=tf.float32, scope="encoder")
    return encoder_output, encoder_final_state

def process_decoder_input(data, vocab2index):
    decoder_input_cut = tf.strided_slice(data, [0,0],[batch_size,-1],[1,1])
    go = tf.fill([batch_size,1],vocab2index["<GO>"])
    return tf.concat([go, decoder_input_cut],1)


#print(fill_data(source_data)[:10])
#print(source_transform[:10])
#print(process_decoder_input(target_transform, target_vocab2index))
def get_encoder_cell(size):
    cell = tf.contrib.rnn.GRUCell(num_units=size)
    return cell

def get_decoder_cell(size, attention_mechanism):
    cell = tf.contrib.rnn.GRUCell(num_units=size)
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=5)
    return attn_cell

def get_decoder(decoder_input, embedding_size, vocab_size, decoder_hidden_size, target_sequence_length, encoder_state, encoder_output):
    #embdding
    decoder_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1,1))
    decoder_input_embedding = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    #decoder
    #decoder_cell = tf.contrib.rnn.GRUCell(num_units=decoder_hidden_size)
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=5, memory=encoder_output)

    decoder_cell =  tf.contrib.rnn.MultiRNNCell([get_decoder_cell(decoder_hidden_size, attention_mechanism) for i in range(3)])
    #decoder_cell =  tf.contrib.rnn.MultiRNNCell([get_encoder_cell(decoder_hidden_size) for i in range(3)])
    
    decoder_output = Dense(len(target_index2vocab))
   
    with tf.variable_scope("decoder"):
        train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_input_embedding, target_sequence_length)
        #not use attention
        #train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, encoder_state, decoder_output)
        #use attention
        train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size), decoder_output)
        train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)

    with tf.variable_scope("decoder", reuse=True):
        predict_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, tf.fill([batch_size], target_vocab2index["<GO>"]), target_vocab2index["<EOS>"])
        #not use attention
        #predict_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predict_helper, encoder_state, decoder_output)
        #use attention
        predict_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predict_helper, decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size), decoder_output)
        predict_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predict_decoder, maximum_iterations=max_length)

    return train_decoder_output, predict_decoder_output
        
        
def get_seq2seq_model(encoder_input, encoder_embedding_size, vocab_size, encoder_hidden_size, 
                      decoder_input, decoder_embedding_size, decoder_hidden_size, target_sequence_length):

    encoder_output, encoder_state = get_encoder(encoder_input, encoder_embedding_size, vocab_size, encoder_hidden_size)
    decoder_input = process_decoder_input(decoder_input, target_vocab2index)
    train_decoder_output, predict_decoder_output = get_decoder(decoder_input, 
                                                               decoder_embedding_size, 
                                                               vocab_size, 
                                                               decoder_hidden_size, 
                                                               target_sequence_length, 
                                                               encoder_state,
                                                               encoder_output)
    return train_decoder_output, predict_decoder_output

    
tf.reset_default_graph()
encoder_embedding_size = 20
decoder_embedding_size = 25
vocab_size = len(source_vocab2index)
encode_hidden_size = 30
decode_hidden_size = 30
batch_size = 100
max_iter_num = 10000
def get_batch(batch_size):
    start = int(np.random.uniform(0, len(source_data_original)-batch_size))
    
    batch_source_transform = source_transform[start:start+batch_size]
    batch_target_transform = target_transform[start:start+batch_size]
    target_seq_length = [len(i) for i in target_data_original[start:start+batch_size]]
    return batch_source_transform, batch_target_transform, target_seq_length

source = tf.placeholder(tf.int32, shape=[batch_size, None])
target = tf.placeholder(tf.int32, shape=[batch_size, None])
seq_length = tf.placeholder(tf.int32, shape=[batch_size])
model_train_output, model_predict_output = get_seq2seq_model(source, 
                                                             encoder_embedding_size, 
                                                             vocab_size, 
                                                             encode_hidden_size, 
                                                             target, 
                                                             decoder_embedding_size, 
                                                             decode_hidden_size, 
                                                             seq_length)

targets = tf.reshape(target, [-1])
logits = tf.reshape(model_train_output.rnn_output, [-1, vocab_size])
loss = tf.losses.sparse_softmax_cross_entropy(targets, logits)
train_op = tf.train.AdamOptimizer().minimize(loss)
predicts = model_predict_output.sample_id

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(max_iter_num):
    batch_source_transform, batch_target_transform, target_seq_length = get_batch(batch_size)
    result = sess.run([train_op,loss,predicts],feed_dict={ source: batch_source_transform,
                                                  target: batch_target_transform,
                                                  seq_length: target_seq_length
                                                })
    print(result[1])
    if i % 2000 == 0:
        print(batch_source_transform[0])
        print(result[2][0])
        print(map_index2word(batch_source_transform[0],source_index2vocab))
        print(map_index2word(result[2][0],target_index2vocab))
        
    


