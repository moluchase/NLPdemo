import tensorflow as tf

def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    """
    construct rnn seq2seq model.
    :param model: model class
    :param input_data: 中传入的数字向量
    :param output_data: 传入的数字向量
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
    cell = cell_fun(rnn_size)
    #cell = cell_fun(rnn_size, state_is_tuple=True)#传入rnn核的大小,当state_is_tuple=True时，状态ct和ht 就是分开记录
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)#堆叠，为两层
    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)

    with tf.device("/cpu:0"):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size, rnn_size], -1.0, 1.0))
        # print("embedding",embedding.shape)
        print("input_data",input_data.shape)
        inputs = tf.nn.embedding_lookup(embedding, input_data)#查找表,将input_data映射到embedding上
        print(inputs.shape)
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)


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