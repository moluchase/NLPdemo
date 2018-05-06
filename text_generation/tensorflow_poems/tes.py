import tensorflow as tf

#产生5*5的均匀分布，在[-1,1]之间
# a=tf.random_uniform([5,5],-1,1)
# with tf.Session() as sess:
#     print(sess.run(a))



embedding=tf.get_variable(name="embedding",initializer=tf.random_uniform([5,5],-1,1))
input=tf.nn.embedding_lookup(embedding,[1,3,4])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(embedding))
    print(sess.run(input))


