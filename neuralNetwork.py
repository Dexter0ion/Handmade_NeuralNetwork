# neural network class definition
# 神经网络类定义
import numpy
import scipy.special


class neuralNetwork:
    # initialise the neural network
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes
        # 设置输入层、隐蔽层、输出层节点数

        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #  link weight matrices,wih and who
        # 连接权重矩阵，weight_input_hidden,weight_hidden_output
        # 正态分布中心点 标准方差 numpy数组大小
        self.wih = numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))

        self.who = numpy.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        # 学习率 误差采用权重
        self.lr = learningrate

        # activation function is the sigmoid function
        # 激励函数是sigmoid函数

        self.activation_function = lambda x: scipy.special.expit(x)
        '''
        Expit ufunc for ndarrays.
        The expit function, also known as the logistic function, 
        is defined as expit(x) = 1/(1+exp(-x)). 
        It is the inverse of the logit function.
        '''

    # train the neural network
    # 训练神经网络

    def train(self, inputs_list, targets_list):
        # convert inputs lists to 2d array
        # 将输入转化为二位数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signal into hidden layer
        # 计算进入隐藏层信号值
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        # 计算隐藏层输出信号（经过sigmoid函数)
        hidden_ouputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_ouputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #error = (target-actual)
        #误差Ek = Tk-Ok

        output_errors = targets - final_outputs

        # calculate hidden errors
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update weight_hidden_output
        self.who += self.lr * \
            numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),
                      numpy.transpose(hidden_ouputs))

        # update weighe_input_hidden
        self.whi += self.lr * \
            numpy.dot((hidden_errors*hidden_ouputs *
                       (1.0-hidden_ouputs)), numpy.transpose(inputs))
        pass

    # query the neural network
    # 检索神经网络
    def query(self, inputs_list):
        # convert inputs lists to 2d array
        # 将输入转化为二位数组
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signal into hidden layer
        # 计算进入隐藏层信号值
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        # 计算隐藏层输出信号（经过sigmoid函数)
        hidden_ouputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_ouputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

        pass
