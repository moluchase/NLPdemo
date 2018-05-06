import collections
import numpy as np

start_token = 'B'
end_token = 'E'


def process_poems(file_name):
    # 诗集
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        #这里处理的不是很好，应该用迭代器的
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')#这里将标题和内容分开
                content = content.replace(' ', '')#去空格
                #这里处理的太粗糙了，按照数据来的吧
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                #这里不知道为什么要控制内容？？？
                if len(content) < 5 or len(content) > 79:
                    continue
                content = start_token + content + end_token#内容的前后都加上标识符
                poems.append(content)
            except ValueError as e:
                pass

    #匿名函数，lambad中的l为传入的参数，len(l)为返回的值
    #关于自定义sorted函数中的key等于一个函数，函数的参数是要比较的list中的元素
    poems = sorted(poems, key=lambda l: len(l))#此处原代码写错了

    #这里的all_words统计的是poems中的每一个字
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    #print('wordNum:'+str(len(all_words)))总共有1721655个字
    counter = collections.Counter(all_words)#print(counter)#是一个dict
    # 因为是升序，而且出现的频率越大，应该越靠前（这个我不是很清楚，应该只要是有序的就可以了吧）
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    #print(count_pairs)
    words, _ = zip(*count_pairs)#返回的是元组，就是将dict转换为两个元组，一个是key的，一个是value的
    #print(words)
    #print(_)

    #print(len(words))#统计后，一共有6109个不同的字
    #这里原代码中并没有截取前面多少个字，我这里取前4999个字再加上空格符
    words =(' ',) +words[:4999]#最后加上空格，表示没有匹配到的，也就是构建的字典中没有的字
    word_int_map = dict(zip(words, range(len(words))))#zip的功能还真强大
    #原代码写的也有问题，取words的长度很显然是不对的，4999该怎么办？
    poems_vector = [list(map(lambda word: word_int_map.get(word, 0), poem)) for poem in poems]
    # for i in range(100):
    #     print(poems[i],poems_vector[i])
    #print(len(words))
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

