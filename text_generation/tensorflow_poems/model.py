import tensorflow as tf

#model:
#input_data
def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    """
    construct rnn seq2seq model.
    :param model: model class
    :param input_data: 之前poems中传入的数字向量
    :param output_data: poems中传入的数字向量
    :param vocab_size:字库大小
    :param rnn_size: rnn核的大小
    :param num_layers: rnn的层数
    :param batch_size: 批次数
    :param learning_rate: 学习率
    :return:
    """
    end_points = {}
    #这里是选择rnn的核
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)#传入rnn核的大小,当state_is_tuple=True时，状态ct和ht 就是分开记录
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)#堆叠，为两层
    #这里分是训练还是测试，对应的初始化state是不同的
    #训练的话，因为是批量训练，因此初始状态的个数应该是batch_size
    #测试的话，也就是输入一个文本，返回一首诗，显然batch_size的大小是1
    #这里返回的是[batch_size, 2*len(cells)],或者[batch_size, s],按照lstm来的，因为由c0和h0组成
    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)

    with tf.device("/cpu:0"):
        #enbedding矩阵，大小为voca_size,rnn_size
        # 其中tf.get_variable的三个参数如下：
        # name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式，初始化的方式有以下几种：
        # tf.constant_initializer：常量初始化函数
        # tf.random_normal_initializer：正态分布
        # tf.truncated_normal_initializer：截取的正态分布
        # tf.random_uniform_initializer：均匀分布
        # tf.zeros_initializer：全部是0
        # tf.ones_initializer：全是1
        # tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size, rnn_size], -1.0, 1.0))
        print("embedding",embedding.shape)
        print("input_data",input_data.shape)
        inputs = tf.nn.embedding_lookup(embedding, input_data)#查找表,将input_data映射到embedding上

    # [batch_size, ?, rnn_size] = [64, ?, 128]
    #RNN网络，输入，初始状态
    #返回的是输出(维度就是[batch_size,max_time,rnn_size])和最后的状态
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)

    print("outputs:",outputs.shape)
    #print("last_state:",last_state[0].shape)
    #-1表示先不看这一维度，先排rnn_size这一维度，把所有的outputs的第二维变成rnn_size,构建二维矩阵
    #参考：https://tensorflow.google.cn/versions/r1.8/api_docs/python/tf/reshape
    output = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)#将bias加到tf.matmul的结果中
    # [?, vocab_size+1]

    if output_data is not None:
        print("output_data:",output_data.shape)
        # 将输出数据（标签）转换为one-hot向量
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size)
        print("labels:",labels.shape)
        #使用交叉熵
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points