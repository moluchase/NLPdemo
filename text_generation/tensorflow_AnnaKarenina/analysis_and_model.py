import os
import random

import numpy as np
import tensorflow as tf
import time

#参数设置
num_steps=100#每个批次的长度（序列长度）
batch_size=64#批次数
embedding_dim=64#词向量维度
lstm_size=128#lstm核大小
num_layers=2#lstm的层数
learning_rate=0.001#学习率
keep_prob=0.8#dropout层中保留的比例
model_dir='./'#模型保存路径
epochs=100#训练的轮数


#将文本数据变为数字数据
#返回的是text变为数字后的list，vocab_map对应表，以及转换后的句子长度
def encodeData(file):
    with open(file,'r') as f:
        text=f.read()
    #n_batchs=int(len(text)/num_steps)#多余部分舍去
    vocab=set(text)#使用set统计单词的个数
    # print(list(vocab),len(vocab))
    vocab_to_int={c:i for i,c in enumerate(vocab)}
    #vocab_map=dict(enumerate(vocab))
    encoded=np.array([vocab_to_int[c] for c in text],dtype=np.int32)
    # print(vocab_to_int[' '])
    return encoded,vocab_to_int,list(vocab)

#生成批次数据
#这里批次取得很不合理感觉，因为没有考虑到断句
def get_batchs(arr):
    stop=int(len(arr)/num_steps)*num_steps
    arr=arr[:stop].reshape((-1,num_steps))
    x_batchs=[]
    y_batchs=[]
    for n in range(0,arr.shape[0],batch_size):
        x=arr[n:n+batch_size if n+batch_size<arr.shape[0] else arr.shape[0]-1,:]#这里注意最后一个批次的选取
        # print(x.shape,',',end='')
        y=np.zeros_like(x)#这里的意思是取x的维度信息，但是值置为0
        y[:,:-1],y[:,-1]=x[:,1:],x[:,0]#这里不知道为什么要把x的第0个元素赋给y的最后一个？？？
        #yield x,y#生成器
        # print(x.shape)
        x_batchs.append(x)
        y_batchs.append(y)
    return x_batchs,y_batchs,len(x_batchs)

#模型构建

def lstm_cell():  # gru核
    return tf.nn.rnn_cell.BasicLSTMCell(lstm_size)


def dropout():  # 为每一个rnn核后面加一个dropout层
    cell = lstm_cell()
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
def model(x,y=None,vocab_size=None):

    res={}#模型返回的结果

    #构建lstm单元
    # lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)#基本单元
    # dropout=tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=keep_prob)#设置dropout
    # cell=tf.nn.rnn_cell.MultiRNNCell([dropout for _ in range(num_layers)])#堆叠
    inputs = tf.one_hot(x, vocab_size)
    with tf.name_scope("rnn"):
        # 多层rnn网络
        cells = [dropout() for _ in range(num_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


        if y is None:
            initial_state = cell.zero_state(1, tf.float32)
            out, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        # out的维度是[batch_size,num_steps,lstm_size],这里我固定了num_steps
        else:
            out, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)  # 这里没有用初始化,但是必须说明dtype


    # embedding = tf.get_variable('embedding', initializer=tf.random_uniform([vocab_size, lstm_size], -1.0, 1.0))#这个代码有点问题,还没有发现,dim只能设置成lstm_size
    # # print("input_data", x.shape)
    # inputs = tf.nn.embedding_lookup(embedding, x)  # 查找表,将input_data映射到embedding上
    # print(inputs.shape)
    #x=tf.reshape(x,[-1])
    # inputs=tf.one_hot(x,vocab_size)
    # if y is None:
    #     initial_state=cell.zero_state(1, tf.float32)
    #     out,last_state=tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state)
    # #out的维度是[batch_size,num_steps,lstm_size],这里我固定了num_steps
    # else: out,_=tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)#这里没有用初始化,但是必须说明dtype

    softmax_w=tf.Variable(tf.truncated_normal([lstm_size,vocab_size],stddev=0.01))
    softmax_b=tf.Variable(tf.zeros(vocab_size))
    out_reshaped=tf.reshape(out,[-1,lstm_size])#前面提到了的，把3维的转为二维，最后一维留下

    #损失
    logits=tf.matmul(out_reshaped,softmax_w)+softmax_b

    if y is not None:
        y_one_hot = tf.one_hot(tf.reshape(y, [-1]), depth=vocab_size)
        print(y_one_hot.shape)
        print(logits.shape)
        #y_shaped = tf.reshape(y_one_hot, [-1, num_steps, vocab_size])
        # y_shaped = tf.reshape(y_one_hot, [-1, vocab_size])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        res['cost']=cost
        res['optimizer']=optimizer
    else:
        predict=tf.nn.softmax(logits)
        res['predict']=predict
        res['last_state']=last_state
        res['initial_state']=initial_state

    return res


def train():
    encoded, vocab_to_int,vocab= encodeData('anna.txt')#文本转数字
    x_batchs, y_batchs , n_batchs= get_batchs(encoded)#获取批次
    vocab_size=len(vocab_to_int)#获取vocab大小

    input_data = tf.placeholder(tf.int32, [None,num_steps], name='input')  # 输入
    output_data = tf.placeholder(tf.int32, [None,num_steps], name='target')  # 输出

    res=model(input_data,output_data,vocab_size=vocab_size)

    saver=tf.train.Saver(max_to_keep=1)#保存模型,参数表示最多保存一个模型
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_epoch=0
        checkpoint = tf.train.latest_checkpoint(model_dir)#取出最近保存的模型
        if checkpoint:
            saver.restore(sess, checkpoint)#将模型赋给sess
            print("## restore from the checkpoint {0}".format(checkpoint))
        print("start training...")
        min_loss=-1
        min_epoch=-1
        try:
            for epoch in range(start_epoch,epochs):
                mean_loss = 0
                for i in range(n_batchs):
                    x,y=x_batchs[i],y_batchs[i]
                    _,loss=sess.run([res['optimizer'],res['cost']],feed_dict={input_data:x,output_data:y})
                    mean_loss+=loss
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, i, loss),time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
                mean_loss=mean_loss/n_batchs
                print('Epoch: %d ,training loss: %.6f' % (epoch, mean_loss),
                      '*' if min_loss<0 or min_loss >mean_loss else '-')
                if min_loss<0 or min_loss>mean_loss:
                    min_loss=mean_loss
                    min_epoch=epoch
                if epoch-min_epoch>5:
                    saver.save(sess, os.path.join(model_dir, 'anna'), global_step=epoch)
                    break

        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            if not tf.train.latest_checkpoint(model_dir):saver.save(sess, os.path.join(model_dir, 'anna'), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))

def generate_text(begin_word):
    encoded, vocab_to_int,vocab= encodeData('anna.txt')  # 文本转数字
    input_data = tf.placeholder(tf.int32, [None, 1],name='input')
    res= model(input_data, None, vocab_size=len(vocab_to_int))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)

        c=begin_word
        if c not in vocab_to_int.keys():
            random.seed()
            c=list[vocab_to_int.keys()][random.randint(0,len(vocab_to_int))]

        x=np.zeros((1,1))
        x[0,0]=vocab_to_int[c]
        predict, last_state = sess.run([res['predict'], res['last_state']],feed_dict={input_data: x})

        text_ = begin_word
        while len(text_)<100:
            c = vocab[np.argmax(predict)]
            text_ += c
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            predict, last_state= sess.run([res['predict'], res['last_state']],
                                             feed_dict={input_data: x, res['initial_state']: last_state})
        return text_


def tes(x,vocab_size):
    # x=tf.one_hot(tf.reshape(x,[-1]),depth=vocab_size)
    # print(x.shape)
    # x_reshaped=tf.reshape(x,[-1,100,vocab_size])
    # print(x_reshaped.shape)
    x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    x_reshaped=tf.reshape(x,[2,3,4])
    #print([[[1,2,3,4],[5,6],[]],[]])
    with tf.Session() as sess:
        print(sess.run(x_reshaped))



# encoded,vocab_map=encodeData('anna.txt')
# x_batchs,y_batchs,n_batchs=get_batchs(encoded)
# print(len(x_batchs),len(y_batchs),n_batchs)
# for i in range(n_batchs):
#     print(i)
    #print(x_batchs[i].shape,y_batchs[i].shape)
#tes(x[0],len(vocab_map))
#print(text_to_int[:200])

train()
#print(generate_text('a'))
#构建词嵌入矩阵的效果 : aI;IpA;40P;M:p4_IpMP;:K;40P;6XPMGIT;:K;40P;6:,pA;:K;40P;6Pp6P;:K;40P;X_:gGpMP;:K;40P;64_IpQP;IpA;40P