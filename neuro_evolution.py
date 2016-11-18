# architecture optimization
# weight optimization 
# hyper parameter optimization (weight initialization?)
# dimensionality reduction

# recurrent networks?
# cellular encoding and ESP

# GA, SA, PS

# Neuro-Evolution

##### can optimize

# shape
# network connection scheme
# weights
# bias(regularization?)
# activation function : linear , logistic, arctan, hyperbolic tangent, M-Logistic, Softmax, Gauss

##### Structured Genetic Algorithm : sGA

# evaluation function : how good are these chromoses(set of parameters not including weights?)? Just train it once and use log loss? train it several times and take the average or best log loss? 
# chromosome : an ordered string of genes
# gene : a 0 or 1, representing a feature of the model
# a chromosome characterizes a generic model : we then apply the evaluation function to the generic model. ex. train the model's weights on training data and check log loss. 
# a phenotype is a particular instance of a chromosome
# 0.) choose a binary string model encoding
# 1.) initialize n chromosomes 
# 2.) evaluate them
# 3.) choose which chromosomes will survive
# 4.) choose how to pair different chromosomes.
# 5.) choose how two chromosomes make a new one ex. one-point crossover + single-point mutation (elitism, passed down genes of top m model don't get mutated)
# 6.) make n babies (how about varying n from generation to generation?)
# 7.) repeat 2 through 7 until generation Z or convergence of evaluation function
# 
# suppose you want to pick the best model with 1-9 hidden layers, where each layer has 1-16 nodes
# the number of layers = XXXX, with the number of nodes in each layer being XXXXX. ofc if a layer doesn't exist, it's XXXXX = 00000
# or you can have each layer have the same amount of nodes, this decreases the search space, but I'm not sure if it's worth it...
# Activation functions, connections and bias can be represented similarly   
# growth rules and cell division 
#
# in combination with simulated annealing : mutation rate fluctuates. Helps jump out of local minima?
#
# two optimization levels : coarse. then manually reset gene encodings based on results of coarse for a fine.
#
# need to make sure that in mating every pair of genes codes the same trait
#
# speciation : write a function that determines whether two networks can mate. allows for multiple groups of very different solutions to develop
# so each species is given x generations to grow to a certain population size, then it enters into competition with the other species?

##### Paralled Distributed Genetic Programming PDGP
# 

##### GeNeralized Acquisition of Recurrent Links GNARL

##### Cellular Encoding CE



##### NEAT
# three principles
# principled crossover : matching genes are inherited 50-50, unmatched genes are inherited only from the fitter parent
# speciation : distance between two organisms exceeds some threshhold then they can't mate. distance defined as linear combination of excess genes, disjoint genes, and shared gene weight differences
# incremental growth from minimum structure : start with a large population of no-hidden layer networks. 
#
# shared fitness : fitness of an organism is divided by the number of organisms in its species. So new species are able to survive?
#
# how are weights updated?... they're mutated. 
#
# speciation is only necessary if GA is applied to topology AND weights
#
# two genes : nodes and connections
# each node is given an id number and is either on or off 
# each connection is given an id number and is either on or off. It also has a weight and the input + output nodes
# a mutation adds a connection between existing nodes, or adds a new node(in neat, adding a new node splits an existing connection(adding depth), this should not always occur, imo.)
# NEAT says this will result in genomes of different sizes. Not necessarily true.


# how to specify arbitrary connections in Neon








# Particle Swarm
#
# 1.) initialize a bunch of solutions, velocities, and inertias
# 2.) for each particle, update its velocity and then its position based on its current velocity, inertia and current performance relative to its personal and the global best
# 3.) do this for x iterations or until convergence


