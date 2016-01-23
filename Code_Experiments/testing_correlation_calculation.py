__author__ = 'Jonathan Simon'

import numpy as np

A = np.random.permutation(12).reshape((4,3))
# A = np.array([1,3,5,4,2,6]).reshape((2,3))
print A, '\n'
o_bar_np = A.mean(0)
A_cent_np = A - o_bar_np # is correct shape for casting?
sigma_np = 1.0/(A.shape[0]-1) * A_cent_np.T.dot(A_cent_np)
inv_std_vec = 1.0/np.sqrt(1.0/(A.shape[0]-1) * (A_cent_np**2).sum(0))
activation_correlation = (inv_std_vec * sigma_np).T * inv_std_vec # works because matrix is symmetric

print "From hand-made function:\n", activation_correlation, '\n'
print "From built-in numpy function:\n", np.corrcoef(A, rowvar=0)
