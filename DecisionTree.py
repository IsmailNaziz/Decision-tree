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
		self.df_train = df.head(int((1-proportion)*df.shape[0]))
		self.df_test = df.head(int(proportion*df.shape[0]))
		self.function = self._init_function(function)
		self.col_type = self._init_col_type(df)
		self.tree = {}


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
		for col in self.df.columns:
			if self.df[col].iloc[0] in [0,1]:
				dic[col]= 'b'
			else:
				dic[col]= 'c'
		return dic

	def select_col(self):
		'''
		x are used to represents ones in a subset 
		y are used to represents zeros in a subset 
		'''
		columns = list(self.df.column)
		dic_func = {}
		for col in columns[:len(columns)-1]:
			dic_func[col] = gini_node(self,col) # returns a tuple 

		ginies_val = [dic_func[key] for key in dic_func.keys()]
		min_value = ginies_val.index(min(ginies_val)) 

		for key, val in dic_func.items():  
			if val == min_value:
				min_key = key

		return dic_func[min_key]


	def gini_binary(self,col,var):
		df_0 = self.df_train[self.df_train['output'] == 0]
		x_0 = df_0[col].sum() # number of ones
		y_0 = df_0.shape[0] - df_1[col].sum() #number of zeros

		df_1 = self.df_train[self.df_train['output'] == 1]
		x_1 = df_1[col].sum() # number of ones
		y_1 = df_1.shape[0] - df_1[col].sum() #number of zeros
	
		p_x_0 = x_0/df_0.shape[0] 
		p_y_0 = y_0/df_0.shape[0] 
		p_x_1 = x_1/df_1.shape[1] 
		p_y_1 = y_1/df_1.shape[1] 

		return((df_0.shape[0]/df_train.shape[0])*self.function(p_x_0,p_y_0) + (df_1.shape[0]/df_train.shape[0])*self.function(p_x_1,p_y_1),var)


	def gini_node(self,col):
		if col_type[col] == 'b':
			return gini_binary(self,col,var)
		else:
			df_list = []
			gini_list = [] 
			self.df_train = df.sort_values(by=[col])
			average_list = [(df_train[col].iloc[i]+df_train[col].iloc[i+1])/2 for i in range(df_train.shape[0]-2)]
			for avg in average_list:
				df_list.append(self.df_train.apply(lambda x : 0 if x <= avg else 1))
			for i in range(df_list):
				 gini_list.append(gini_binary(self,col,average_list[i]))
			
			ginies_val = [gini_list[i][0] for i in range(len(gini_list))]
			min_index = ginies_val.index(min(ginies_val))
			return gini_list[min_index]


if __name__ == '__main__':
	df = pd.read_csv('D:\machine_learning\generated_data\data_4.csv',index_col=False)
	df.drop(df.columns[0], axis=1, inplace=True)
	df['output'] = df['output'].apply(lambda x : 0 if x < 0.5 else 1)	
	print(df)