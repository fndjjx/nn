from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)


import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32, [None,10])
loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
tfarg = tf.argmax(y,1)
correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct,tf.float32))

sess = tf.Session()
with sess.as_default(): 
    tf.global_variables_initializer().run()
    for i in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        train_step.run({x:batch_x,y_:batch_y})

    print(acc.eval({x:mnist.test.images,y_:mnist.test.labels}))
    print(correct.eval({x:mnist.test.images,y_:mnist.test.labels}))
    print(tfarg.eval({x:mnist.test.images}))
    print(y.eval({x:mnist.test.images}))
    print(y_.eval({y_:mnist.test.labels}))
