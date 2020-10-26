# -*- coding: utf-8 -*-
"""
@author: Naziz Ismail 
"""


import pandas as pd
import numpy as np
from Node import Node
import copy

class DecisonTree(object):
	"""docstring for DecisonTree"""

	def __init__(self,df,function,proportion = 0.3):
		df_train = df.head(int((1-proportion)*df.shape[0]))
		df_test = df.head(int(proportion*df.shape[0]))
		self.function = self._init_function(function)
		self.col_type = self._init_col_type(df)
		self.first_node = self._init_first_node(self.df_train) # init
		self.meta_graph = {} # key : parent , value : child classic graph
		self.mapping_meta_graph = {} # key : key metagraph, value :  ('col', None if binary treshhold if continuous)


	def _init_first_node(self):
		node = select_node(self.df_train)
		


	def _init_function(self,function):
		if function == 'Gini':
			return self.gini()
		else:
			return self.entropy()

	@classmethod
	def gini(cls,x,y): 
		'''
		x and y are 2 probabilities 

		'''
		return 1 - x**2 - y**2

	@classmethod
	def entropy(cls,x,y):
		'''
		x and y are 2 probabilities 
		
		'''
		return - np.log(x)*x - np.log(y)*y

	def _init_col_type(self):
		dic = {}
		for col in df.columns:
			if df[col].iloc[0] in [0,1]:
				dic[col]= 'b'
			else:
				dic[col]= 'c'
		return dic

	def generate_df(df,node):
		node_tool = node.copy()
		df_tool = df.copy()
		depth = 0
		path = []
		while node_tool.prev != None:
			depth+=1
			path = node_tool.success
			node_tool = node_tool.prev

		if depth == 0:
			return df 

		for i in range(depth-1):
			if node_tool.next_succ != None:
				res = (1 if path[i] ==True else 0)
				res_browse = (node_tool.next_succ if path[i] ==True else node_tool.next_fail)
				df_tool = node_tool.split_node(df_tool)[res]
				node_tool = res_browse
			else: 
				df_tool = node_tool.split_node(df_tool)[res]

		return df_tool

	def step(df,node):
		# return the vale for the next node
		df = generate_df(df,node) # generate df for that step
		min_node = select_node(df)
		if min_node.func_val < node.fun_val:
			node.next = min_node 
			return min_node 

		return False

	def create_tree(self):
		res_step = step(self.df_train,self.first_node)

		if step(node) != false:


		return None 
		

	def select_node(df):
	
		columns = list(df.columns)
		dic_func = {}
		for col in columns[:len(columns)-1]:
			L_features = self.func_node(col) # returns a Node 

		return self.min_node_list(L_features)

 	
	def min_node_list(L_nodes):
		'''
		return node with minimum of fun_val in list of nodes
		'''
		mini = L_nodes[0].func_val
		index = 0
		for i in range(len(L_nodes)):
			if L_nodes[i].func_val < mini:
				mini = L_nodes[i].func_val
				index = i

		return L_nodes[i]



	def func_binary(df,col,treshhold = 0.5):
		df_0 = df[df['output'] == 0]
		x_0 = df_0[col].sum() # number of ones
		y_0 = df_0.shape[0] - df_1[col].sum() #number of zeros

		df_1 = df[df['output'] == 1]
		x_1 = df_1[col].sum() # number of ones
		y_1 = df_1.shape[0] - df_1[col].sum() #number of zeros
	
		p_x_0 = x_0/df_0.shape[0] 
		p_y_0 = y_0/df_0.shape[0] 
		p_x_1 = x_1/df_1.shape[1] 
		p_y_1 = y_1/df_1.shape[1] 

		func_val = (df_0.shape[0]/df.shape[0])*self.function(p_x_0,p_y_0) + (df_1.shape[0]/df.shape[0])*self.function(p_x_1,p_y_1)

		return Node(col,func_val,treshhold)


	def func_node(df,col):

		'''
		x are used to represents ones in a subset 
		y are used to represents zeros in a subset 
		'''

		if self.col_type[col] == 'b':
			return self.func_binary(col,treshhold)
		else:
			df_list = []
			val_func_list = [] 
			df = df.sort_values(by=[col])

			# Tranformation to be compatible with the func_binary
			average_list = [(df[col].iloc[i]+df[col].iloc[i+1])/2 for i in range(df.shape[0]-1)]
			for avg in average_list:
				df_list.append(df[col].apply(lambda x : 0 if x <= avg else 1))

			#apply fun_binary
			for i in range(len(df_list)):
				 val_func_list.append(self.func_binary(df_list[i],col,average_list[i]))
			
			return self.min_node_list(val_func_list)


if __name__ == '__main__':
	df = pd.read_csv('D:\machine_learning\generated_data\data_5_IM_OB.csv',index_col=False)
	df.drop(df.columns[0], axis=1, inplace=True)
	print(df)