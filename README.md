# Inception: compressible neural networks with faster convergence
A normal neural network layer has the form of sigmoid(A\*X + B). This experiment sets A = A1\*A2 + A3 and B = B1\*B2 + B3 so that the layer is now of the form sigmoid((A1\*A2 + A3)\*X + B1\*B2 + B3). This form results in faster convergence, and the terms (A1\*A2 + A3) and (B1\*B2 + B3) can be compressed down to matrices after training.

## Results
This idea has been tested on two problems: xor function and the iris dataset. In repeated experiments the xor function is learned ~8x faster and the iris dataset is learned ~4x faster. The faster learning rate does come at considerable computational cost. The increased rate is likely the result of an interplay between the momentum filters and the network weights.

Below are example traces of the cost vs epochs for xor and iris:

### XOR epochs
![epochs of xor](cost_xor.png?raw=true)

### Iris epochs
![epochs of iris](cost_iris.png?raw=true)
