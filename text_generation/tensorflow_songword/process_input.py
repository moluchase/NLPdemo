import collections
import numpy as np




def process_words(file_name):
    # 宋词
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for i,line in enumerate(f.readlines()):
            try:
                poems.append(line[:-1])
            except ValueError as e:
                pass

    #这里的all_words统计的是poems中的每一个字
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    #print('wordNum:'+str(len(all_words)))#总共有1690977
    counter = collections.Counter(all_words)#print(counter)#是一个dict
    #print(counter)
    #这里我就想知道
    # num=0
    # for i,j in counter.items():
    #     if j>2:num+=1
    # print(num)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    #print(count_pairs)
    words, _ = zip(*count_pairs)#返回的是元组，就是将dict转换为两个元组，一个是key的，一个是value的
    #print(len(words))
    words =(' ',) +words[:4999]#最后加上空格，表示没有匹配到的，也就是构建的字典中没有的字
    word_int_map = dict(zip(words, range(len(words))))#zip的功能还真强大
    poems_vector = [list(map(lambda word: word_int_map.get(word, 0), poem)) for poem in poems]
    return poems_vector, word_int_map, words#返回数字向量，字映射表，字典V，实际这里不需要返回words的

#传入的参数是batch_size，数字向量，字映射表
def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = int(len(poems_vec)/batch_size)#分批数，这里相当于余数不要了，不过如果batch_size比较小的话，可以不要
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vec[start_index:end_index]
        length = max(map(len, batches))#取最长的作为这些向量的长度
        # 不够的用‘’补齐，这里之前文本分类使用的是固定长度600，此处长度是诗的长度，使用固定长度不好
        #具体做法是先生成一个全部是' '对应的数字矩阵，然后再填充
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]#此处输出数据和输入数据是错位的，输出的最有一个元素不变，可以看出rnn的结构了
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches

#process_words('宋词.txt')