import pandas as pd


# 先由CART得到一颗决策树
train_data_1 = pd.read_csv('data/CV-3/train_dealt_1.csv')
train_data_2 = pd.read_csv('data/CV-3/train_dealt_2.csv')
train_data = pd.concat([train_data_1, train_data_2], axis=0)
Y_data = train_data['Survived']
X_data = train_data.drop('Survived', axis=1)
tree = DecisionTreeCART(X_data, Y_data).root_node

# 剪枝算法
k = 0
alpha = float('inf')
T_child = list()
T = tree
T_child.append(T)