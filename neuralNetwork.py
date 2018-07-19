# neural network class definition
# 神经网络类定义


class neuralNetwork:
    # initialise the neural network
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes
        # 设置输入层、隐蔽层、输出层节点数

        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        # 学习率 误差采用权重

        self.lr = learningrate

    # train the neural network
    # 训练神经网络
    def train():
        pass

    # query the neural network
    # 检索神经网络
    def query():
        pass
