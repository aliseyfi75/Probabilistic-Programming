import time
from joblib import Parallel, delayed

def f(n):
	for i in range(n):
		for j in range(n):
			k = i + j

x = 70

start_0 = time.time()
[f(i**2) for i in range(x)]
print("non-parallel version time: {}".format(time.time() - start_0))

start_1 = time.time()
Parallel(n_jobs=1)(delayed(f)(i**2) for i in range(x))
print("parallelized (1 job) version time: {}".format(time.time() - start_1))

start_2 = time.time()
Parallel(n_jobs=2)(delayed(f)(i**2) for i in range(x))
print("parallelized (2 job) version time: {}".format(time.time() - start_2))

start_3 = time.time()
Parallel(n_jobs=3)(delayed(f)(i**2) for i in range(x))
print("parallelized (3 job) version time: {}".format(time.time() - start_3))

start_4 = time.time()
Parallel(n_jobs=4)(delayed(f)(i**2) for i in range(x))
print("parallelized (4 job) version time: {}".format(time.time() - start_4))
