
# tf.nn.bias_add(value, bias, name=None)
# 解释：这个函数的作用是将偏差项bias加到value上面。
# 这个操作你可以看做是tf.add的一个特例，其中bias必须是一维的。该API支持广播形式，因此
# value可以有任何维度。但是，该API又不像tf.add可以让bias的维度和value的最后一维不同，tf.nn.bias_add中bias的维度和value最后一维必须相同。
# 输入参数：
# value: 一个Tensor。数据类型必须是float，double，int64，int32，uint8，int16，int8或者complex64。
# bias: 一个一维的Tensor，数据维度和
# value
# 的最后一维相同。数据类型必须和value相同。
# name: （可选）为这个操作取一个名字。
# 输出参数：
# 一个Tensor，数据类型和value相同。

import tensorflow as tf
a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
b = tf.constant([2.0, 1.0])
c = tf.constant([1.0])
sess = tf.Session()
print(sess.run(tf.nn.bias_add(a, b)))
# 因为 a 最后一维的维度是 2 ，但是 c 的维度是 1，所以以下语句将发生错误
print(sess.run(tf.nn.bias_add(a, c)))
# 但是 tf.add() 可以正确运行
print(sess.run(tf.add(a, c)))


# 获取变量维度是一个使用频繁的操作，在tensorflow中获取变量维度主要用到的操作有以下三种：
#
#     Tensor.shape
#     Tensor.get_shape()
#     tf.shape(input,name=None,out_type=tf.int32)
#
# 对上面三种操作做一下简单分析：（这三种操作先记作A、B、C）
#
#     A 和 B 基本一样，只不过前者是Tensor的属性变量，后者是Tensor的函数。
#     A 和 B 均返回TensorShape类型，而 C 返回一个1D的out_type类型的Tensor。
#     A 和 B 可以在任意位置使用，而 C 必须在Session中使用。
#     A 和 B 获取的是静态shape，可以返回不完整的shape； C 获取的是动态的shape，必须是完整的shape。
