# -*- coding: utf-8 -*-
"""
@author: Naziz Ismail 
"""


import pandas as pd
import numpy as np


class Node(object):
	"""docstring for Node"""

	def __init__(self,feature,func_val,treshhold,next_succ = None,next_fail = None,prev = None,success = None):
		self.feature = feature
		self.func_val = func_val
		self.treshhold = treshhold
		self.next_succ = next_succ
		self.next_fail = next_fail
		self.prev = prev
		self.success = success
		self.res_series = self._init_plit_series() # for lambda calcul
		self.split_node = self._init_plit_df()

	def _init_plit_series(self):
		def f(x):
			if x > treshhold :
				return True 
			else:
				return False

	def _init_plit_df(self):
		def f(df):
			return df[df[self.feature] > treshhold],df[df[self.feature] <= treshhold]


	