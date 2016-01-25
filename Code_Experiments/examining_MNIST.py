__author__ = 'Jonathan Simon'

import cPickle
import gzip

f = gzip.open("data/mnist.pkl.gz", 'rb')

# Each of the three datasets is a tuple of length 2, containing the x-values (as numpy arrays) in the first
# component, and the y-values (as numpy ints) in the second component.
# 'train_set' contains 50k samples, 'valid_set' contains 10k samples, 'test_set' contains 10k samples
# The samples within each set are already randomized (i.e. labeled are not in any particular order)
train_set, valid_set, test_set = cPickle.load(f)
f.close()

print train_set[0].shape
# (50000, 784)
print train_set[1].shape
# (50000,)

print valid_set[0].shape
# (10000, 784)
print valid_set[1].shape
# (10000,)

print test_set[0].shape
# (10000, 784)
print test_set[1].shape
# (10000,)