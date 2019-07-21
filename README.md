# Inception: compressible neural networks with faster convergence
A normal neural network layer has the form of sigmoid(A\*X + B). This experiment sets A = A<sub>1</sub>\*A<sub>2</sub> + A<sub>3</sub> and B = B<sub>1</sub>\*B<sub>2</sub> + B<sub>3</sub> so that the layer is now of the form sigmoid((A<sub>1</sub>\*A<sub>2</sub> + A<sub>3</sub>)\*X + B<sub>1</sub>\*B<sub>2</sub> + B<sub>3</sub>). This form results in faster convergence, and the terms (A<sub>1</sub>\*A<sub>2</sub> + A<sub>3</sub>) and (B<sub>1</sub>\*B<sub>2</sub> + B<sub>3</sub>) can be compressed down to matrices after training.

## Results
This idea has been tested on two problems: xor function and the iris dataset. In repeated experiments the xor function is learned ~8x faster and the iris dataset is learned ~4x faster. The faster learning rate does come at considerable computational cost. The increased rate is likely the result of an interplay between the momentum filters and the network weights.

Below are example traces of the cost vs epochs for xor and iris:

### XOR epochs
![epochs of xor](cost_xor.png?raw=true)

### Iris epochs
![epochs of iris](cost_iris.png?raw=true)
