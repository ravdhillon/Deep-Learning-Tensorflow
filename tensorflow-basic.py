import tensorflow as tf
x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.scalar_mul(x1, x2)
print(result)

with tf.compat.v1.Session() as sess:
    output = sess.run(result)
    print(output)
