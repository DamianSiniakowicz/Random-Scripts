# check bindings before adding blas

# numpy.show_config()

# did i use sudo ldconfig right?

# we are using LAPACK and BLAS, no ATLAS
# we want to use gfortran, not g77
# One relatively simple and reliable way to check for the compiler used to build a library is to use ldd on the library. If libg2c.so is a dependency, this means that g77 has been used. If libgfortran.so is a a dependency, gfortran has been used. If both are dependencies, this means both have been used, which is almost always a very bad idea.

# i used git clone and git checkout to build numpy, will any files I add to numpy be overwritten when I sudo apt-get update numpy?


# use ldd on the openblas .so file in my venv numpy to make sure i'm hooked up to gfortran


# before custom setup
'''
>>> numpy.show_config()
lapack_opt_info:
    libraries = ['openblas', 'openblas']
    library_dirs = ['/usr/local/lib']
    define_macros = [('HAVE_CBLAS', None)]
    language = c
blas_opt_info:
    libraries = ['openblas', 'openblas']
    library_dirs = ['/usr/local/lib']
    define_macros = [('HAVE_CBLAS', None)]
    language = c
openblas_info:
    libraries = ['openblas', 'openblas']
    library_dirs = ['/usr/local/lib']
    define_macros = [('HAVE_CBLAS', None)]
    language = c
openblas_lapack_info:
    libraries = ['openblas', 'openblas']
    library_dirs = ['/usr/local/lib']
    define_macros = [('HAVE_CBLAS', None)]
    language = c
blas_mkl_info:
  NOT AVAILABLE

>>> numpy.__config__.openblas_info
{'libraries': ['openblas', 'openblas'], 'library_dirs': ['/usr/local/lib'], 'define_macros': [('HAVE_CBLAS', None)], 'language': 'c'}



'''






# how can I set the command line argument as the value of a variable in my script
# specifically, as a string representing the file I want to write to 


import numpy as np
import numpy.random as npr
import time
import pickle

output_path = 'my_data/with_blas.pkl'

speed_dict = {'Matrix_Multiplication' : [], 'Dot_Product' : [], 'SVD' : [], 'Eigen_Decomposition' : []}

for run in range(100):
 
	# --- Test 1
	N = 1
	n = 1000
	 
	A = npr.randn(n,n)
	B = npr.randn(n,n)
	 
	t = time.time()
	for i in range(N):
	    C = np.dot(A, B)
	td = time.time() - t
	print("dotted two (%d,%d) matrices in %0.1f ms" % (n, n, 1e3*td/N))
	speed_dict['Matrix_Multiplication'].append(td)
 
	# --- Test 2
	N = 100
	n = 4000
	 
	A = npr.randn(n)
	B = npr.randn(n)
	 
	t = time.time()
	for i in range(N):
	    C = np.dot(A, B)
	td = time.time() - t
	print("dotted two (%d) vectors in %0.2f us" % (n, 1e6*td/N))
	speed_dict['Dot_Product'].append(td)	
 
	# --- Test 3
	m,n = (2000,1000)
	 
	A = npr.randn(m,n)
	 
	t = time.time()
	[U,s,V] = np.linalg.svd(A, full_matrices=False)
	td = time.time() - t
	print("SVD of (%d,%d) matrix in %0.3f s" % (m, n, td))
	speed_dict['SVD'].append(td)	
 
	# --- Test 4
	n = 1500
	A = npr.randn(n,n)
	 
	t = time.time()
	w, v = np.linalg.eig(A)
	td = time.time() - t
	print("Eigendecomp of (%d,%d) matrix in %0.3f s" % (n, n, td))
	speed_dict['Eigen_Decomposition'].append(td)

pickle.dump( speed_dict, open( output_path, 'wb'))

