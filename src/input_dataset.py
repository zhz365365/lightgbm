import lightgbm as lgb
import numpy as np
class DatasetTest:
	def __init__(self):
		self._matrix1 = lgb.Dataset('../data/train.svm.txt')
		self._matrix2 = lgb.Dataset(data=np.arange(0, 12).reshape((4, 3)), label=[1, 2, 3, 4],
									weight=[0.5, 0.4, 0.3, 0.2], 
									silent=False, feature_name=['a', 'b', 'c'])
	def print_matrix(self, matrix):
		print('data: %s' % matrix.data)
		print('label: %s' % matrix.label)
		print('weight: %s' % matrix.weight)
		print('init_score: %s' % matrix.init_score)
		print('group: %s' % matrix.group)
	def run_method(self, matrix):
		print('get_ref_chain():', matrix.get_ref_chain(ref_limit=10))
		print('subset():', matrix.subset(used_indices=[0,1]))
	def test(self):
		self.print_matrix(self._matrix1)
		# data: data/train.svm.txt
		# label: None
		# weight: None
		# init_score: None
		# group: None

		self.print_matrix(self._matrix2)
		# data: [[ 0  1  2]
		#  [ 3  4  5]
		#  [ 6  7  8]
		#  [ 9 10 11]]
		# label: [1, 2, 3, 4]
		# weight: [0.5, 0.4, 0.3, 0.2]
		# init_score: No

		self.run_method(self._matrix2)

if __name__ == '__main__':
	Dataset = DatasetTest()
	Dataset.test()
