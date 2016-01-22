# Correlation-Penalized-Networks
Research on the effects of decorrelating the hidden activations in neural networks

Initially testing the behavior of several possible alterations:

1.  covariance penalty (fixed weight)
2.  correlation penalty (fixed weight)
3.  covariance penalty (decaying weight)
4.  correlation penalty (decaying weight)

Additional steps will include:

- testing multiple weight and decay paramater weights
- testing with both the theano tutorial MLP and Convnet
- altering minibatch size
- sampling from unit activations
- sampling from minibatch samples
- randomizing minibatches
