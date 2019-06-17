# Beeline gradient descent

Gradient descent with shared weights.
The number of shared weights is increased with each epoch, so the dimensionality of the search space is increased with each epoch.

Below is a base cost curve for the iris data set:

![epochs of normal gradient descent](epochs_normal.png?raw=true)

Cost curve for beeline gradient descent:

![epochs of beeline gradient descent](epochs_shared.png?raw=true)

As you can see beeline gradient descent still converges, but doesn't work as well as the normal approach.
