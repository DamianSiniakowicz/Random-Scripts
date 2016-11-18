#!/usr/bin/env python





from numpy import genfromtxt
'''
import sys
import click 

@click.command()
@click.option('--All/--no-All', default=False)


def info(All):
    if All:
        print "not implimented"
    	return(7)
    else:
	train_data = genfromtxt('numerai_data_5/numerai_training_data.csv',delimiter=',',skip_header=1)
	tournament_data = genfromtxt('numerai_data_5/numerai_tournament_data.csv',delimiter=',',skip_header=1) 
	tournament_data_x = tournament_data[:,1:22]
	tournament_id = tournament_data[:,0]
	
info()
'''

# set up the datasets

train_data = genfromtxt('/home/damian/Dropbox/numerai/Week_8/numerai_training_data.csv',delimiter=',',skip_header=1)
tournament_data = genfromtxt('/home/damian/Dropbox/numerai/Week_8/numerai_tournament_data.csv',delimiter=',',skip_header=1) 
tournament_data_x = tournament_data[:,1:22]
tournament_id = tournament_data[:,0]


import numpy as np

global inversions
inversions = 0

def slow_change(previous_error=9001,current_error=1000):
	print(previous_error)
	print(current_error)
	if abs(previous_error - current_error) < .000000001:
		return((current_error,True))
	elif (previous_error - current_error) < 0.0:
		global inversions
		inversions += 1
		print("INVERSIONS!")
		print(inversions)
		if inversions > 10:
			return((current_error,True))
		else:
			return((current_error,False))
	else:
		global inversions
		inversions = 0		
		return((current_error,False)) 

average_train_cost = 0
average_test_cost = 0

all_train_costs = []
all_test_costs = []

for run in range(1):


	np.random.shuffle(train_data)
	'''
	train_data_x = train_data[0:80000,0:21]
	train_data_y = train_data[0:80000,21]
	test_data_x = train_data[80000:96321,0:21]
	test_data_y = train_data[80000:96321,21]
	from neon.util.argparser import NeonArgparser
	parser = NeonArgparser(__doc__)
	args = parser.parse_args()
	from neon.data import ArrayIterator
	train_data_y = train_data_y.astype(int)
	test_data_y = test_data_y.astype(int)
	train = ArrayIterator(X=train_data_x,y=train_data_y,nclass=2)
	test = ArrayIterator(X=test_data_x,y=test_data_y,nclass=2)
	'''
	train_x = train_data[:,0:21]
	train_y = tain_data[:,21]
	train_x = train_x.astype(int)
	train_y = train_y.astype(int)	
	from neon.util.argparser import NeonArgparser
	parser = NeonArgparser(__doc__)
	args = parser.parse_args()
	from neon.data import ArrayIterator
	TRAIN = ArrayIterator(X=train_x,y=train_y)
	TOURNEY = ArrayIterator(X = tournament_data_x)

	# initialize weights
	from neon.initializers import Gaussian
	init_norm = Gaussian(loc=0.0, scale=0.1)
	# layers defines how each output layer is connected to the next. transforms defines the function each node applies to the raw data it is given.
	from neon.layers import Affine, Dropout
	from neon.transforms import Logistic
	# set up the weight matrices (connection strengths) and the neuron's processing function
	layers = []
	#layers.append(Dropout())
	layers.append(Affine(nout=50, init=init_norm, activation=Logistic()))
	layers.append(Dropout())
	layers.append(Affine(nout=40, init=init_norm, activation=Logistic()))
	layers.append(Dropout())
	layers.append(Affine(nout=30, init=init_norm, activation=Logistic()))
	layers.append(Dropout())
	layers.append(Affine(nout=20, init=init_norm, activation=Logistic()))
	layers.append(Dropout())
	layers.append(Affine(nout=10, init=init_norm, activation=Logistic()))
	layers.append(Dropout())
	layers.append(Affine(nout=2, init=init_norm, activation=Logistic()))
	'''
	layers.append(Affine(nout=50, init=init_norm, activation=Logistic()))
	layers.append(Dropout())
	layers.append(Affine(nout=50, init=init_norm, activation=Logistic()))
	layers.append(Dropout())
	layers.append(Affine(nout=50, init=init_norm, activation=Logistic()))
	layers.append(Dropout())
	layers.append(Affine(nout=50, init=init_norm, activation=Logistic()))
	layers.append(Dropout())
	layers.append(Affine(nout=50, init=init_norm, activation=Logistic()))
	layers.append(Dropout())'''	
	
	# set up the model
	from neon.models import Model
	mlp = Model(layers=layers)
	# set up a cost function 
	from neon.layers import GeneralizedCost
	from neon.transforms import CrossEntropyBinary, SumSquared, LogLoss
	cost = GeneralizedCost(costfunc=CrossEntropyBinary())
	# optimizer
	from neon.optimizers import RMSProp, GradientDescentMomentum
	optimizer = RMSProp()
	# progress bar
	from neon.callbacks.callbacks import Callbacks, MetricCallback, EarlyStopCallback, SerializeModelCallback, DummyClass,  CollectWeightsCallback
	# callbacks = Callbacks(model = mlp  , eval_set = train, eval_freq = 1, output_file = "model_stats/some_data_1")
	callbacks = Callbacks(mlp, eval_set=TRAIN, eval_freq = 1)
	#callbacks.add_callback(SerializeModelCallback('right_here.data',1,10))
	#callbacks.add_callback(CollectWeightsCallback('weight_history'))
	callbacks.add_early_stop_callback(slow_change)	
	# callbacks.add_callback(MetricCallback(train,metric = LogLoss()))	
	# callbacks.add_callback(MetricCallback(eval_set=train, metric=LogLoss(), epoch_freq=1))
	# callbacks.add_callback(LossCallback(eval_set=test, epoch_freq=1))
	# fit the model
	mlp.fit(TRAIN, optimizer=optimizer, num_epochs=100000000, cost=cost, callbacks=callbacks) # what is this args.epochs business?
	# test the model on test set
	results = mlp.get_outputs(TRAIN)
	# evaluate the model on test_set using the log loss metric
	train_error = mlp.eval(TRAIN, metric=LogLoss())
	average_train_cost += train_error
	#test_error = mlp.eval(test, metric=LogLoss())
	#average_test_cost += test_error
	all_train_costs.append(train_error)
	#all_test_costs.append(test_error)
	print('Train Log Loss = %f' % train_error)
	#print('Test Log Loss = %f' % test_error)

print('Average Train Log Loss = %f' % average_train_cost)
#print('Average Test Log Loss = %f' % average_test_cost)

#master_list = {'all train costs' : all_train_costs,'all test costs' : all_test_costs,'average train cost' : average_train_cost,'average test cost' : average_test_cost}

import numpy

tourney_results = mlp.get_outputs(TOURNEY)

preds = numpy.array([tournament_id,tourney_results])

#import pickle

#pickle.dump( master_list, open( "single_dataset", "wb" ) )

#from neon.visualizations.custom_viz import *

#layer_train_speed('weight_history','master_plot_so_great')


# QUESTION
# why did cross binary entropy error balloon to 17 when I used a 21 by 1500 by 2 network?

'''
# TO DO

# purpose : evaluate a model? 

# print train and validation costs for each epoch

# plot learning curves

# record training time


# TO DO

# automate everything

# automate architecture selection

# automate hyperparameter selection

# find better optimization algorithms

'''


'''
	layers.append(Affine(nout=10, init=init_norm, activation=Logistic()))
	layers.append(Affine(nout=10, init=init_norm, activation=Logistic()))
	layers.append(Affine(nout=10, init=init_norm, activation=Logistic()))
	layers.append(Affine(nout=10, init=init_norm, activation=Logistic()))
	layers.append(Affine(nout=10, init=init_norm, activation=Logistic()))
'''





















