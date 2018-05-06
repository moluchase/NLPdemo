import random

import tensorflow as tf
from tensorflow_songword.model import rnn_model
from tensorflow_songword.process_input import process_words
import numpy as np


model_dir = './'
corpus_file = './宋词.txt'

lr = 0.0002

#这个难道不是随机产生一个数
def to_word(predict, vocabs):
    sample=np.argmax(predict)
    return vocabs[sample]#返回的是下标对应的字


def gen_poem(begin_word):
    batch_size = 1
    print('## loading corpus from %s' % model_dir)
    poems_vector, word_int_map, vocabularies = process_words(corpus_file)#返回数字向量，字映射表，字典V

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='gru', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=256, num_layers=2, batch_size=64, learning_rate=lr)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)

        word=begin_word
        if word not in word_int_map.keys():
            random.seed()
            word=list[word_int_map.keys()][random.randint(0,5000)]

        x=np.zeros((1,1))
        x[0,0]=word_int_map[word]
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})

        poem_ = begin_word+''
        while word.strip(' '):
            word = to_word(predict, vocabularies)
            poem_ += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            if len(poem_)>5 and poem_[:-4].find(poem_[-4:-1])>0:
                poem_=poem_[:-4]
                break
        return poem_


def pretty_print_poem(poem_):
    poem_sentences = poem_.split('。')
    for s in poem_sentences:
        if s.strip(' ').strip('，'):print(s + '。')

if __name__ == '__main__':
    begin_char = input('## please input the first character:')
    poem = gen_poem(begin_char)
    pretty_print_poem(poem_=poem)