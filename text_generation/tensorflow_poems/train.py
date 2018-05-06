import os
import tensorflow as tf
from tensorflow_poems.model import rnn_model
from tensorflow_poems.poems import process_poems, generate_batch

#方便命令行输入
#三个参数：变量名，默认参数，描述
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('model_dir', os.path.abspath('./'), 'model save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./poems.txt'), 'file name of poems.')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS


def run_training():
    # if not os.path.exists(FLAGS.model_dir):
    #     os.makedirs(FLAGS.model_dir)

    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)#这里默认V的大小是5000
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)#创建一个saver，然后获取全部变量并保存
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())#合并
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        #下面这几行的目的就是接着前面的start_epoch继续训练
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)#到模型保存路径中获取最近一次保存的模型
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        max_loss=-1
        has_num=0
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = int(len(poems_vector)/FLAGS.batch_size)#批次数
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    if max_loss<0:max_loss=loss
                    elif max_loss>loss:
                        max_loss=loss
                        has_num=0
                    else:has_num+=1
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss),'*' if max_loss==loss else '-')
                if has_num>10:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
                    break

        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            if not tf.train.latest_checkpoint(FLAGS.model_dir):saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))



if __name__ == '__main__':
    run_training()