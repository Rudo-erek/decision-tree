from decisionTree import DecisionTree
import pandas as pd

import random as rnd

import traceback

def inorder_traversal(treeNode):
    childNodes = treeNode.children
    if childNodes is None or len(childNodes) == 0:
        print(treeNode.category + '\n')
        return
    for key in childNodes:
        if childNodes[key].feature is None:
            print("key:" + key + " category:" + str(childNodes[key].category) + "\n")
        else:
            print("key:" + key + " feature:" + childNodes[key].feature + "\n")
            inorder_traversal(childNodes[key])
    return

def test_decision_tree(Y_test, tree):
    count = 0
    Y_test['Survived'] = None
    nums = Y_test.shape[0]
    for i in range(nums):
        row = Y_test.loc[i]
        find_category(row, tree)
        Y_test.loc[i, 'Survived'] = row['Survived']
        # count += 1
        # print(count)
    print('done')

def find_category(row, treeNode):
    childNodes = treeNode.children
    if treeNode.feature is None:
        row['Survived'] = treeNode.category
        return
    else:
        node = childNodes.get(row[treeNode.feature])
        if node is None:
            row['Survived'] = rnd.randint(0, 1)
            return
        else:
            find_category(row, node)
            return

if __name__ == '__main__':
    train_data = pd.read_csv(r'E:\learn\Machine_Learning_course\paper&code\decision_tree\data\train_dealt.csv')
    # X_data = train_data.drop('haogua', axis=1)
    # Y_data = train_data['haogua']
    Y_data = train_data['Survived']
    X_data = train_data.drop('Survived', axis=1)
    tree = DecisionTree(X_data, Y_data).root_node

    # Y_test = pd.read_csv(r'E:\learn\Machine_Learning_course\paper&code\decision_tree\data\test_dealt.csv')
    # Y_test.to_csv(r'E:\learn\Machine_Learning_course\paper&code\decision_tree\data\test_dealt_out.csv', index=False)
    
    
    Y_test = pd.read_csv(r'E:\learn\Machine_Learning_course\paper&code\decision_tree\data\train_dealt_test.csv')
    test_decision_tree(Y_test, tree)
    Y_test.to_csv(r'E:\learn\Machine_Learning_course\paper&code\decision_tree\data\train_dealt_test_out.csv', index=False)
    
