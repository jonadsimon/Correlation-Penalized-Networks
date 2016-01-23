__author__ = 'Jonathan Simon'

'''
Experimenting with the Theano 'scan' function
'''

# import timeit
import numpy as np
import theano
# import theano.tensor as T

# start_time = timeit.default_timer()

# A = np.array([range(15)]).reshape((5,3))
# A = np.array([range(4)]).reshape((2,2))
A = np.array([1,3,5,4,2,6]).reshape((2,3))
print A, '\n'
o_bar_np = A.mean(0)
print o_bar_np, '\n'
A_cent_np = A - o_bar_np # is correct shape for casting?
print A_cent_np, '\n'
sigma_np = 1.0/(A.shape[0]-1) * A_cent_np.T.dot(A_cent_np)
print sigma_np, '\n\n'


A_theano = theano.shared(value=A, name='A_theano', borrow=True)
print A_theano.eval(), '\n'
mean_activation = A_theano.mean(0)
print mean_activation.eval(), '\n'
centered_activation = A_theano - mean_activation # casts over rows
print centered_activation.eval(), '\n'
activation_covariance = 1.0/(A_theano.shape[0]-1) * centered_activation.T.dot(centered_activation)
print activation_covariance.eval(), '\n'
off_diag_cov_sqr = (activation_covariance**2).sum() - (activation_covariance**2).diagonal().sum()
print off_diag_cov_sqr.eval(), '\n\n'

# # np_cov = np.zeros((7,7))
# np_cov = 0
# for i in range(X.shape[0]):
#     np_cov += X[[i]].T.dot(X[[i]])
#
# print np_cov, '\n'
#
# # Flattens rows...
# # for row in X:
# #     np_cov += row.T.dot(row)
#
#
# X2 = T.matrix('X2')
# # result, updates = theano.scan(fn=lambda prior_result,row: prior_result+row.T.dot(row), outputs_info=[T.zeros((X2.shape[1],X2.shape[1]))], sequences=[X2])
# result, updates = theano.scan(fn=lambda row: row.shape_padaxis(0).dot(row.shape_padaxis(1)), sequences=[X2])
# # final_result = result[-1]
# # theano_cov = theano.function(inputs=[X2], outputs=final_result, updates=updates)
# # theano_cov2 = theano.function(inputs=[X2], outputs=result, updates=updates)
# theano_cov_sub = theano.function(inputs=[X2], outputs=result, updates=updates)
#
# print theano_cov_sub(X)

# print theano_cov(X)
# print
# print theano_cov2(X)

# [[ 2.  3.]
#  [ 2.  5.]]
#
# [[[ 0.  1.]
#   [ 0.  1.]]
#
#  [[ 2.  3.]
#   [ 2.  5.]]]