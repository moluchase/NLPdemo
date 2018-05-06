## poems.py
分两个函数
### process_poems(file_name)
主要是将文本转为数字，具体的先统计txt文件（2万首诗）中各个字出现的频率，这里总共有6千多个字，我取了前5000个作为字
然后就是建立映射表，5000个字中每个字都对应着一个数字
关于python中的zip
将字典放到zip中，就变成两个元组了
```
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)
```
将两个list放到zip中，再加上dict就变成dict了
```
word_int_map = dict(zip(words, range(len(words))))
```
### generate_batch(batch_size, poems_vec, word_to_int)
这部分是批量数据生成，主要就是将poems_vec进行分批处理，其中y标签是x标签的错位结果
从数据可以看出此处文本生成使用的是简单的RNN模型，也就是第i个输入的输出是第i+1个的输入

## model.py
采用的是两层，核大小是128，embedding矩阵的维度和ｒｎｎ核大小相同

## train.py
这一部分就是调用ｍｏｄｅｌ了，然后训练

## compose_poems.py
因为是文本生成，也就是传入一个字，然后由该字生成一首诗
代码中先使用开始符'B'生成一个字，然后判断是否给出了字，如果给出，就用该字进行生成，
因为是一步一步的（即t=1），这里代码中也是一步步执行的，也就是将上一步的y作为当前步的输入，上一步的隐含状态作为当前的初始状态
这里我有点不太懂：feed_dict中传入的参数形式应该是怎么样的，之前以为只能是placeholder，不作更新，而且传入的值和里面的值是如何对应的。。。