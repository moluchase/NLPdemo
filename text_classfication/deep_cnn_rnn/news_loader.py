from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr


#将文件中的标签和内容进行分类
def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename,'r') as f:
        #每一条新闻都是一个标题和内容组成
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels

#构建字表
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """
    :param train_dir: 训练内容
    :param vocab_dir: 字表存储目录
    :param vocab_size: 字表大小
    :return: void
    """

    data_train, _ = read_file(train_dir)#这里只需要内容
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)#统计字频
    count_pairs = counter.most_common(vocab_size - 1)#统计出现最多的4999个字,这里还有一个未出现字
    words, _ = list(zip(*count_pairs))#返回两个list,分别是key和value
    words = ['<PAD>'] + list(words)#这里字表并没有按照频率进行排序
    with open(vocab_dir, mode='w')as f:
        f.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """

    :param vocab_dir: 字表目录
    :return: words是字表集,word_to_id是一个map,字和数字之间的对应关系
    """
    with open(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """
    有10个分类,构建分类id
    :return: 类别,分类和数字之间的关系
    """
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


"""将id表示的内容转换为文字"""
def to_words(content, words):
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """

    :param filename: 文件名
    :param word_to_id: 字和数字的对应表
    :param cat_to_id: 标签和数字的对应表
    :param max_length: 最大长度,这里将每篇文章取前600个有效字
    :return: x_pad表示的是一条文本,用数字表示,而且长度固定,y_hot是一个one_hot向量
    """
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

#批次大小为64
def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]