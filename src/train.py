import lightgbm as lgb
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

_label_map={
	'Iris-setosa':0,
	'Iris-versicolor':1,
	'Iris-virginica':2,
}
class BoosterTest:
	def __init__(self):
		df = pd.read_csv('../data/iris.csv')
		_feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
		x = df[_feature_names]
		y = df['Class'].map(lambda x: _label_map[x])

		train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.3, stratify=y, shuffle=True, random_state=1)
		self._train_set = lgb.Dataset(data=train_X, label=train_Y, feature_name=_feature_names)
		self._validate_set = lgb.Dataset(data=test_X, label=test_Y, reference=self._train_set)
		self._booster = lgb.Booster(
			params={
				'boosting':'gbdt',
				'verbosity':1,  # 打印消息
				'learning_rate':0.1,  # 学习率
				'num_leaves':7,
				'min_data':2,
				'application':'multiclass',
				'num_class':3,
				'metric':['multi_error', 'multi_logloss'],
				'early_stopping_round':5,
				'seed':321,
			},
			train_set=self._train_set,
		)
		self._booster.add_valid(self._validate_set,'validate')
		self._booster.set_train_data_name('train')
	def print_attr(self):
		print('feature name:',self._booster.feature_name())
		# feature name: ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
		print('feature nums:', self._booster.num_feature())
		# feature nums: 4
	def test_train(self):
		for i in range(0, 10):
			self._booster.update(self._train_set)
			print('after iter:%d'%self._booster.current_iteration())
			print('train eval:',self._booster.eval(self._train_set, name='train'))
			print('test eval:',self._booster.eval(self._validate_set, name='eval'))
		# after iter:1
		# train eval: [('train', 'auc', 0.9776530612244898, True)]
		# test eval: [('eval', 'auc', 0.9783333333333334, True)]
		# after iter:2
		# train eval: [('train', 'auc', 0.9907142857142858, True)]
		# test eval: [('eval', 'auc', 0.9872222222222222, True)]
		# after iter:3
		# train eval: [('train', 'auc', 0.9922448979591837, True)]
		# test eval: [('eval', 'auc', 0.9888888888888889, True)]
		# after iter:4
		# train eval: [('train', 'auc', 0.9922448979591837, True)]
		# test eval: [('eval', 'auc', 0.9888888888888889, True)]
	def test(self):
		self.print_attr()
		self.test_train()

class TrainTest:
	def __init__(self):
		df = pd.read_csv('../data/iris.csv')
		_feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
		x = df[_feature_names]
		y = df['Class'].map(lambda x: _label_map[x])
		train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.3, stratify=y, shuffle=True, random_state=1)
		self._train_set = lgb.Dataset(data=train_X, label=train_Y, feature_name=_feature_names)
		self._validate_set = lgb.Dataset(data=test_X, label=test_Y, reference=self._train_set)
	def train_test(self):
		params={
			'boosting':'gbdt',
			'verbosity':1,  # 打印消息
			'learning_rate':0.1,  # 学习率
			'num_leaves':7,
			'min_data':2,
			'application':'multiclass',
			'num_class':3,
			'metric':['multi_error', 'multi_logloss'],
			'early_stopping_round':5,
			'seed':321,
		}
		eval_rst = {}
		gbm = lgb.train(params, self._train_set, num_boost_round=20,
						valid_sets=[self._train_set, self._validate_set],
						valid_names=['valid1', 'valid2'],
						early_stopping_rounds=5, evals_result=eval_rst, verbose_eval=True)
		print('eval_rst:', eval_rst)
		lgb.plot_importance(gbm)
		plt.tight_layout()
		plt.savefig('../visual/importance.png')
		plt.close()
	def cv_test(self):
		params={
			'boosting':'gbdt',
			'verbosity':1,  # 打印消息
			'learning_rate':0.1,  # 学习率
			'num_leaves':7,
			'min_data':2,
			'application':'multiclass',
			'num_class':3,
			'metric':['multi_error', 'multi_logloss'],
			'early_stopping_round':5,
			'seed':321,
		}
		eval_history = lgb.cv(params, self._train_set, num_boost_round=20, nfold=3, stratified=True,
							  early_stopping_rounds=5, verbose_eval=True, shuffle=True)
		print('eval_history:', eval_history)

class SKLTest:
	def __init__(self):
		df = pd.read_csv('../data/iris.csv')
		_feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
		x = df[_feature_names]
		y = df['Class'].map(lambda x: _label_map[x])
		self.train_X, self.test_X, self.train_Y, self.test_Y = \
			train_test_split(x, y, test_size=0.3, stratify=y, shuffle=True, random_state=1)
	def train_test(self):
		clf = lgb.LGBMClassifier(boosting_type='gbdt',
								 num_leaves=15,
								 min_data=2,
								 learning_rate=0.1,
								 n_estimators=100,
								 objective='multiclass',
								 num_class=3,
								 njobs=7,
								 silent=False)
		clf.fit(self.train_X, self.train_Y, eval_metric='logloss', eval_set=[(self.test_X, self.test_Y),],
				early_stopping_rounds=5)
		print('evals_results:', clf.evals_result_)
		print('predict:', clf.predict(self.test_X))

if __name__ == '__main__':
	Booster = BoosterTest()
	Booster.test()
	Train = TrainTest()
	Train.train_test()
	Train.cv_test()
	SKL = SKLTest()
	SKL.train_test()
