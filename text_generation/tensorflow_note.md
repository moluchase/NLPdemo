## tensorflow 函数
### tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
如果我希望是input传入这个cell时dropout掉一部分input信息的话，就设置input_keep_prob,那么传入到cell的就是部分input；
如果我希望这个cell的output只部分作为下一层cell的input的话，就定义output_keep_prob。
我的理解是input_keep_prob是设置输入的，特指第一层输入，而output_keep_prob是设置输出的，指的是层与层之间的

### tensorflow中的name_scope, variable_scope
https://www.zhihu.com/question/54513728/answer/177724045

### 关于tf.nn.rnn_cell和tf.contrib.rnn
是等价的,但是前者将取代后者

### 关于enbedding矩阵维度和rnn核相同
```python
lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)#基本单元
dropout=tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=keep_prob)#设置dropout
cell=tf.nn.rnn_cell.MultiRNNCell([dropout for _ in range(num_layers)])#堆叠
embedding = tf.get_variable('embedding', initializer=tf.random_uniform([vocab_size, lstm_size], -1.0, 1.0))#这个代码有点问题,还没有发现,dim只能设置成lstm_size
inputs = tf.nn.embedding_lookup(embedding, x)  # 查找表,将input_data映射到embedding上
out,_=tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)
```
上面这个,可以看到嵌入矩阵的维度大小等于lstm的核大小,但是如果不相等的话,是会报错的
后面我找了好久,再网上查到需要将这块代码放到一个变量域中,但是不知道为什么
比如下面这个代码,one_hot大小和lstm核大小就不一样
```python
def lstm_cell():  # gru核
    return tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
def dropout():  # 为每一个rnn核后面加一个dropout层
    cell = lstm_cell()
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
def model(x,y=None,vocab_size=None):
    res={}#模型返回的结果
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
```
### 关于feed_dict
自己写tensorflow_AnnaKarenina代码后,才基本搞懂了,
一般情况下,我的model函数参数是这样的
```python
def model(x,y=None,vocab_size=None)
  ...
  return optimizer
```
然后我在train中调用该model,获取optimizer来执行,如下:
```python
input = tf.placeholder(tf.int32, [None,num_steps], name='input')  # 输入
output = tf.placeholder(tf.int32, [None,num_steps], name='target')  # 输出
res=model(input,output,vocab_size=vocab_size)
```
我先使用的是占位符,然后下面执行的时候再传入变量,这样的好处是不会新增很多常量节点
```python
for i in range(n_batchs):
  x,y=x_batchs[i],y_batchs[i]
  _,loss=sess.run([res['optimizer'],res['cost']],feed_dict={input:x,output:y})
  mean_loss+=loss
```
可以看到占位符的名字和model中的参数是对应的,我们先将占位符传入模型,占位符名也和feed_dict冒号左边的名字是对应的,此时左边的变量就是model中的一个变量,右边的值表示将该值赋值给model中的变量

那么我们也可以给model中已有的变量进行传值,比如下面:
model中如下
```python
def model(x,y=None,vocab_size=None):
  ...
  predict=tf.nn.softmax(logits)
  res['predict']=predict
  res['last_state']=last_state
  res['initial_state']=initial_state
```
调用的时候,这个last_state是上一次调用model后返回的res['last_state'],将其传入到feed_dict中,赋值给model已有的变量res['initial_state']
```python
predict, last_state= sess.run([res['predict'], res['last_state']],
                                             feed_dict={input_data: x, res['initial_state']: last_state})
```