import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5],
          [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0],
          [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Measure Accuracy
predicted = tf.argmax(hypothesis, 1)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(predicted, tf.argmax(Y, 1)), dtype=tf.float32))

# Cross Entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})

        if step % 20 == 0:
            cost_val = sess.run(cost, feed_dict={X: x_data, Y: y_data})
            print("step:", step, "cost: ", cost_val)

    print("Accuracy:", sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))