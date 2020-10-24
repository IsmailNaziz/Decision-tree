# -*- coding: utf-8 -*-
"""
@author: Naziz Ismail 
"""


import pandas as pd
import numpy as np
import copy


class DecisonTree(object):
	"""docstring for DecisonTree"""

	def __init__(self,df,function,proportion = 0.3):
		df_train = df.head(int((1-proportion)*df.shape[0]))
		df_test = df.head(int(proportion*df.shape[0]))
		self.function = self._init_function(function)
		self.col_type = self._init_col_type(df)
		self.meta_graph = {} # key : parent , value : child classic graph
		self.mapping_meta_graph = {} # key : key metagraph, value :  ('col', None if binary treshhold if continuous)


	def create_tree(self,df):

		res = self.select_col(df)
		self.mapping_meta_graph[i] = res
		self.meta_graph[i] = (i+1,i+2)


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

	def select_col(df):
	
		columns = list(df.columns)
		dic_func = {}
		for col in columns[:len(columns)-1]:
			dic_func[col] = self.func_node(col) # returns a tuple (val_fun,treshhold or None for binary )

		fun_vals = [dic_func[key] for key in dic_func.keys()]
		min_value = fun_vals.index(min(fun_vals)) 

		for key, val in dic_func.items():  
			if val == min_value:
				key_min = key

		return dic_func[key_min]

 
	def func_binary(df,col,var = None):
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

		return((df_0.shape[0]/df.shape[0])*self.function(p_x_0,p_y_0) + (df_1.shape[0]/df.shape[0])*self.function(p_x_1,p_y_1),var,col)


	def func_node(df,col):

			'''
			x are used to represents ones in a subset 
			y are used to represents zeros in a subset 
			'''

		if self.col_type[col] == 'b':
			return self.func_binary(col,var)
		else:
			df_list = []
			val_func_list = [] 
			df = df.sort_values(by=[col])

			# Tranformation to be compatible with the func_binary
			average_list = [(df[col].iloc[i]+df[col].iloc[i+1])/2 for i in range(df.shape[0]-1)]
			for avg in average_list:
				df_list.append(df[col].apply(lambda x : 0 if x <= avg else 1))
			for i in range(len(df_list)):
				 val_func_list.append(self.func_binary(df,col,average_list[i]))
					
			# selection of the average that minimizes loss function
			fun_vals = [val_func_list[i][0] for i in range(len(val_func_list))]
			min_index = fun_vals.index(min(fun_vals))

			return val_func_list[min_index]


if __name__ == '__main__':
	df = pd.read_csv('D:\machine_learning\generated_data\data_5_IM_OB.csv',index_col=False)
	df.drop(df.columns[0], axis=1, inplace=True)
	print(df)