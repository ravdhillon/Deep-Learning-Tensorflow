import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("/tmp/data",  one_hot=True)
# one_hot -- Used for multi class classification. We have 10 classes (0-9)
# one_hot means only single entry(one pixel) in the output class list is ON and rest are OFF.


#1. Define your model parameters like classes, batch size, etc.
epochs = 10
n_classes = 10
batch_size = 128

# Images size is 28 x 28.
chunk_size = 28
n_chunks = 28
rnn_size = 128

#3. Placeholders
x = tf.placeholder('float', [None, n_classes, chunk_size])
y = tf.placeholder('float')

#4. Define Neural Network
def recurrent_neural_network_model(x):
    # input_data * weights + biases
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, state = rnn.rnn(lstm_cell, x, dtype=tf.float32)


    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output

#5. Train Neural Network
def train_neural_network(x):
    prediction = recurrent_neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # Got the cost, now we need to minimize the cost. It will do that by applying a learning rate.
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # training phase
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, ' completed out of ', epochs, 'loss:', epoch_loss)
        
        
        # ArgMax
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)