class linearRegression:
    """最小二乘法"""

    def fit(self, X, y):
        # 换成矩阵
        X = np.asmatrix(X.copy())
        y = np.asmatrix(y).reshape(-1, 1)

        # 通过最小二乘公式求出最佳权重
        self.w_ = (X.T * X).I * X.T * y

    def predict(self, X):
        """对样本数据进行预测 """
        # 将X转换成矩阵
        X = np.asmatrix(X.copy())
        result = X * self.w_
        # 将矩阵转换成ndarray数组，进行扁平化处理ravel()
        return np.asarray(result).ravel()

        # 设截距的为w0，增加一列，作为第一列
        t = boston.sample(len(boston), random_state=0)
        new_columns = t.columns.insert(0, "Intercept")
        t = t.reindex(columns=new_columns, fill_value=1)

        # 训练集测试集划分 0.8左右
        train_X = t.iloc[:400, :-1]
        train_y = t.iloc[:400, -1]
        test_X = t.iloc[400:, :-1]
        test_y = t.iloc[400:, -1]

        lr = linearRegression()
        lr.fit(train_X, train_y)
        result = lr.predict(test_X)
        # 查看方差
        display(np.mean((result - test_y) ** 2))
        # 查看模型权重
        display(lr.w_)


class LinearRegression2:
    """梯度下降法"""

    def __init__(self, alpha, times):
        # alpha : 学习率，用来控制步长（权重调整幅度）
        # times :  循环迭代的次数
        self.alpha = alpha
        self.times = times

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        # 初始权重为0
        self.w_ = np.zeros(1 + X.shape[1])
        # 损失值
        self.lose_ = []
        # 进行循环，多次迭代，使得损失值不断减小
        for i in range(self.times):
            # 计算预测值

            # np.dot(),计算点积
            y_hat = np.dot(X, self.w_[1:]) + self.w_[0]
            # 计算真实值与预测值之间的差距
            error = y - y_hat
            # 将损失加入到损失列表中
            self.lose_.append(np.sum(error ** 2) / 2)
            # 根据差值，调整权重self.w_，公式 权重(j) = 权重(j) + 学习率alpha * sum((y - y_hat) * x(j))
            self.w_[0] += self.alpha * np.sum(error)
            self.w_[1:] += self.alpha * np.dot(X.T, error)

    def predict(self, X):
        # 对样本进行预测

        X = np.asarray(X)
        result = np.dot(X, self.w_[1:]) + self.w_[0]
        return result

