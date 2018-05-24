from treeNode import TreeNode
import pandas as pd
import numpy as np


class DecisionTree:

    def __init__(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        self.root_node = TreeNode(None, None, None, None, self.X_train, self.Y_train)
        self.features = self.get_features(self.X_train)
        self.tree_generate(self.root_node)

    def get_features(self, X_train_data):
        features = dict()
        for i in range(len(X_train_data.columns)):
            feature = X_train_data.columns[i]
            features[feature] = list(X_train_data[feature].value_counts().keys())

        return features

    def tree_generate(self, tree_node):
        X_data = tree_node.X_data
        Y_data = tree_node.Y_data
        # get all features of the data set
        features = list(X_data.columns)
        if len(list(Y_data.value_counts())) == 1:
            tree_node.category = Y_data.iloc[0]
            tree_node.children = None
            return
        elif len(features) == 0:
            tree_node.category = Y_data.value_counts(ascending=False).keys()[0]
            tree_node.children = None
            return
        else:
            ent_d = self.compute_entropy(Y_data)
            XY_data = pd.concat([X_data, Y_data], axis=1)
            d_nums = XY_data.shape[0]
            max_gain_ratio = 0
            feature = None

            for i in range(len(features)):
                v = self.features.get(features[i])
                # v = list(X_data[features[i]].value_counts().keys())
                Ga = ent_d
                IV = 0
                for j in v:
                    dv = XY_data[XY_data[features[i]] == j]
                    dv_nums = dv.shape[0]
                    ent_dv = self.compute_entropy(dv[dv.columns[-1]])
                    Ga -= dv_nums/d_nums*ent_dv
                    # IV -= dv_nums/d_nums*np.log2(dv_nums/d_nums)

                # if (Ga/IV) > max_gain_ratio:
                if Ga > max_gain_ratio:
                    # max_gain_ratio = Ga/IV
                    max_gain_ratio = Ga
                    feature = features[i]

            if feature is None:
                feature = features[0]
            tree_node.feature = feature

            # get all kinds of values of the current partition feature
            branches = self.features.get(feature)
            # branches = list(XY_data[feature].value_counts().keys())
            tree_node.children = dict()
            for i in range(len(branches)):
                X_data = XY_data[XY_data[feature] == branches[i]]
                if len(X_data) == 0:
                    category = XY_data[XY_data.columns[-1]].value_counts(ascending=False).keys()[0]
                    childNode = TreeNode(tree_node, None, None, category, None, None)
                    tree_node.children[branches[i]] = childNode
                    # return
                    # error, not should return, but continue
                    continue

                Y_data = X_data[X_data.columns[-1]]
                X_data.drop(X_data.columns[-1], axis=1, inplace=True)
                X_data.drop(feature, axis=1, inplace=True)
                childNode = TreeNode(tree_node, None, None, None, X_data, Y_data)
                tree_node.children[branches[i]] = childNode
                # print("feature: " + str(tree_node.feature) + " branch: " + str(branches[i]) + "\n")
                self.tree_generate(childNode)

            return

    def compute_entropy(self, Y):
        ent = 0;
        for cate in Y.value_counts(1):
            ent -= cate*np.log2(cate);
        return ent




