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

Standard MLP takes ~9.6s per iteration (with batch_size=20)
Correlation-penalized MLP takes ~42.9s per iteration (with batch_size=20)
Therefore correlation-penalized MLP takes ~4.5x longer to run
~~Covariance-Penalized MLP takes ~27.4s per iteration~~
~~Therefore Covariance-Penalization takes ~2.85x longer to run~~

Note that increasing the batch size should increase the usefulness of the penalization, by providing a more accurate estimate of each hidden unit’s activation time course

Decided to use *entire* covariance matrix for penalization, since main diagonal is always constant = 1, and therefore including it does not alter the gradient (I checked).