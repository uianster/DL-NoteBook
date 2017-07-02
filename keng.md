# tensorflow 爬过的坑


##  2.sparse_softmax_cross_entropy_with_logit和softmax_cross_entropy_with_logitsque 的区别。
之前一直被误导，以为sparse_softmax_cross_entropy_with_logit的便签也是one-hot形式。于是一直报错如下：
***ValueError: Rank mismatch: Rank of labels (received 2) should equal rank of logits minus 1 (received 1).***

后来看了 stackoverflow后才明白，原来sparse_softmax_cross_entropy_with_logits输入的就是putu
Both functions computes the same results and sparse_softmax_cross_entropy_with_logits computes the cross entropy directly on the sparse labels instead of converting them with one-hot encoding.

You can verify this by running the following program:

import tensorflow as tf
from random import randint

dims = 8
pos  = randint(0, dims - 1)

logits = tf.random_uniform([dims], maxval=3, dtype=tf.float32)
labels = tf.one_hot(pos, dims)

res1 = tf.nn.softmax_cross_entropy_with_logits(       logits=logits, labels=labels)
res2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.constant(pos))

with tf.Session() as sess:
    a, b = sess.run([res1, res2])
    print a, b
    print a == b
