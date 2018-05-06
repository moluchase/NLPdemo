import os
import tensorflow as tf
from tensorflow_songword.model import rnn_model
from tensorflow_songword.process_input import process_words, generate_batch

#方便命令行输入
#三个参数：变量名，默认参数，描述
batch_size= 64
learning_rate= 0.01
model_dir='./'
file_path='./宋词.txt'
model_prefix='words'
epochs=50




def run_training():
    poems_vector, word_to_int, vocabularies = process_words(file_path)#这里默认V的大小是5000
    batches_inputs, batches_outputs = generate_batch(batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='gru', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=256, num_layers=2, batch_size=64, learning_rate=learning_rate)

    saver = tf.train.Saver(tf.global_variables())#创建一个saver，然后获取全部变量并保存
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())#合并
    with tf.Session() as sess:
        sess.run(init_op)

        start_epoch = 0
        #下面这几行的目的就是接着前面的start_epoch继续训练
        checkpoint = tf.train.latest_checkpoint(model_dir)#到模型保存路径中获取最近一次保存的模型
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        min_loss=-1
        loss_index=0
        try:
            for epoch in range(start_epoch, epochs):
                n = 0
                n_chunk = int(len(poems_vector)/batch_size)#批次数
                mean_loss=0
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    mean_loss+=loss
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))
                cur_loss=mean_loss/n_chunk
                if min_loss<0 or cur_loss<min_loss:
                    loss_index=epoch
                    min_loss=cur_loss
                    saver.save(sess, os.path.join(model_dir, model_prefix), global_step=epoch)
                print('Epoch: %d,training loss: %.6f' % (epoch, loss),'*' if epoch==loss_index else '-')
                if epoch-loss_index>3:break

        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(model_dir, model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))



if __name__ == '__main__':
    run_training()