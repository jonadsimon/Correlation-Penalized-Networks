# Correlation-Penalized-Networks
Research on the effects of decorrelating the hidden activations in neural networks

Initially testing the behavior of several possible alterations:

1.  covariance penalty (fixed weight)
2.  correlation penalty (fixed weight)
3.  covariance penalty (decaying weight)
4.  correlation penalty (decaying weight)

Each of the above will be tested with several different weight and decay paramater settings. Each of the above will be tested on both the theano tutorial MLP and Convnet.

Additional experiments may include altering the minibatch size, and sampling from the activations in the minibatch.
