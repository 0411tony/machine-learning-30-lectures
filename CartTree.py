#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

lst = ['a', 'b', 'c', 'd', 'b', 'c', 'a', 'b', 'c', 'd', 'a']
def gini(nums):
    probs = [nums.count(i)/len(nums) for i in set(nums)]
    gini = sum([p*(1-p) for p in probs])
    return gini
gini(lst)

df = pd.read_csv('./example_data.csv', dtype={'windy': 'str'})
gini(df['play'].tolist())


def split_dataframe(data, col):
    unique_values = data[col].unique()
    result_dict = {elem: pd.DataFrame for elem in unique_values}
    for key in result_dict.keys():
        result_dict[key] = data[:][data[col] == key]
    return result_dict

def choose_best_col(df, label):
    gini_D = gini(df[label].tolist())
    cols = [col for col in df.columns if col not in [label]]
    min_value, best_col = 999, None
    min_splited = None
    for col in cols:
        splited_set = split_dataframe(df, col)
        gini_DA = 0
        for subset_col, subset in splited_set.items():
            gini_Di = gini(subset[label].tolist())
            gini_DA += len(subset)/len(df)*gini_Di
        
        if gini_DA < min_value:
            min_value, best_col = gini_DA, col
            min_splited = splited_set
    return min_value, best_col, min_splited

class CartTree:
    class Node:
        def __init__(self, name):
            self.name = name
            self.connections = {}
        
        def connect(self, lable, node):
            self.connections[label] = node
        
    def __init__(self, data, label):
        self.columns = data.columns
        self.data = data
        self.label = label
        self.root = self.Node("Root")
    
    def print_tree(self, node, tabs):
        print(tabs + node.name)
        for connection, child_node in node.connections.items():
            print(tabs + "\t" "(" + connection +")" )
            self.print_tree(child_node, tabs+"\t\t")
    
    def construct_tree(self):
        self.construct(self.root, "", self.data, self.columns)
    
    def construct(self, parent_node, parent_connection_label, input_data, columns):
        min_value, best_col, min_splited = choose_best_col(input_data[columns], self.label)
        if not best_col:
            node = self.Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label, node)
            return
        node = self.Node(best_col)
        parent_node.connect(parent_connection_label, node)
        
        new_columns = [col for col in columns if col != best_col]
        for splited_value, splited_data in min_splited.items():
            self.construct(node, splited_value, splited_data, new_columns)
tree1 = CartTree(df, 'play')
tree1.construct_tree()
tree1.print_tree(tree1.root, "")
