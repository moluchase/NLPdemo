# NLPdemo
NLP方面的一些小的demo,包括文本生成,文本分类,等等,使用tensorflow实现,长期更新,欢迎指正,交流
## text_generation
这部分有3个demo,分别是唐诗生成,宋词生成,AnnaKarenina文本生成

这里先给出参考文章或代码以及比较好的文章：

[基于Char-RNN Language Model进行文本生成（Tensorflow生成唐诗](https://blog.csdn.net/Irving_zhang/article/details/76664998)

[中文古诗自动作诗机器人](https://github.com/jinfagang/tensorflow_poems)

[好玩的文本生成](https://www.msra.cn/zh-cn/news/features/ruihua-song-20161226)

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[《安娜卡列尼娜》文本生成——利用TensorFlow构建LSTM模型](https://zhuanlan.zhihu.com/p/27087310)

[Neural Text Generation: A Practical Guide](https://www-cs.stanford.edu/~zxie/textgen.pdf)

[文本生成概述](https://www.jiqizhixin.com/articles/2017-05-22)

### 唐诗生成

**poems.py**
分两个函数

**process_poems(file_name)**

主要是将文本转为数字，具体的先统计txt文件（2万首诗）中各个字出现的频率，这里总共有6千多个字，我取了前5000个作为字 然后就是建立映射表，5000个字中每个字都对应着一个数字 关于python中的zip 将字典放到zip中，就变成两个元组了
```
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)
```
将两个list放到zip中，再加上dict就变成dict了

```
word_int_map = dict(zip(words, range(len(words))))
```

**generate_batch(batch_size, poems_vec, word_to_int)**

这部分是批量数据生成，主要就是将poems_vec进行分批处理，其中y标签是x标签的错位结果 从数据可以看出此处文本生成使用的是简单的RNN模型，也就是第i个输入的输出是第i+1个的输入

**model.py**

采用的是两层，核大小是128，embedding矩阵的维度和rnn核大小相同【这里是有问题的，如果不一样会报错，是因为变量域的问题，我在总的notes中给出了解释】

**train.py**

这一部分就是调用ｍｏｄｅｌ了，然后训练

**compose_poems.py**

因为是文本生成，也就是传入一个字，然后由该字生成一首诗 代码中先使用开始符'B'生成一个字，然后判断是否给出了字，如果给出，就用该字进行生成， 因为是一步一步的（即t=1），这里代码中也是一步步执行的，也就是将上一步的y作为当前步的输入，上一步的隐含状态作为当前的初始状态 

这里：feed_dict中传入的参数形式应该是怎么样的，之前以为只能是placeholder，不作更新，而且传入的值和里面的值是如何对应的。。。【总的notes中也给出了解释】


### 宋词生成

基本上就是模仿poems的代码写的 这里全是宋词，我在网上爬下来的，大概有两万多首 然后我没有用开始词和结束词，有以下几点原因 在之前的代码中如果使用了'B'，在诗生成的过程中，先是通过'B'来生成隐含状态，也就是说第一个字和‘B'是没有什么关系的 对于最后使用'E'，我觉得也没有什么用，因为最后都是'。'，也就是'。'后面都是'E'，那我们学习的是什么呢？ 这里使用的rnn_size变为了256，因为句子可能会很长。 我在生成代码中就先用给定的字先走一步，也就是feed_dict中只传入input_data，而隐含状态就直接在model里初始化 然后得到了此步的隐含状态和预测值，然后进入到循环中，feed_dict中传入预测字和上一步的隐含状态 在生成过程中，进行了一些小的处理，详见代码

最后我使用'小'，生成如下词
```
小院深深，风雨引晴，天气潋，春风满院，满院帘栊。
正是是，春光正好，春色渐成阴。
春光渐觉，渐夭砂、夭砂黄吸。 
嫩黄嫩，嫩蛟嫩润，渐渐。
```

### 《安娜卡列尼娜》文本生成

这个效果特别差，不知道代码哪里出问题了,调试中...

## text_classfication

这是我最早做的一个demo，也是参考别人的代码，后期会做一个整理

