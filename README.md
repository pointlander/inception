# Inception: compressible neural networks with faster convergence
A normal neural network layer has the form of sigmoid(A\*X + B). This experiment sets A = A<sub>1</sub>\*A<sub>2</sub> + A<sub>3</sub> and B = B<sub>1</sub>\*B<sub>2</sub> + B<sub>3</sub> so that the layer is now of the form sigmoid((A<sub>1</sub>\*A<sub>2</sub> + A<sub>3</sub>)\*X + B<sub>1</sub>\*B<sub>2</sub> + B<sub>3</sub>). This form results in faster convergence, and the terms (A<sub>1</sub>\*A<sub>2</sub> + A<sub>3</sub>) and (B<sub>1</sub>\*B<sub>2</sub> + B<sub>3</sub>) can be compressed down to matrices after training.

## Results
This idea has been tested on two problems: xor function and the iris dataset. In repeated experiments the xor function is learned ~10x faster and the iris dataset is learned ~10x faster. The faster learning rate does come at considerable computational cost. The increased rate is likely the result of an interplay between the optimizer and the network weights.

### XOR Experiments
| Mode | Optimizer | Converged | Epochs |
| ---- | --------- | --------- | ------ |
| inception | static | 0.941406 | 230.319502 |
| inception | momentum | 0.949219 | 271.798354 |
| inception | adam | 0.937500 | 422.245833 |
| normal | momentum | 0.976562 | 3157.240000 |
| normal | static | 0.972656 | 3480.598394 |
| normal | adam | 0.855469 | 4882.771689 |

### Iris Experiments
| Mode | Optimizer | Converged | Epochs |
| ---- | --------- | --------- | ------ |
| inception | momentum | 1.000000 | 206.394531 |
| inception | static | 1.000000 | 213.019531 |
| inception | adam | 1.000000 | 557.566406 |
| normal | momentum | 1.000000 | 3565.992188 |
| normal | static | 1.000000 | 3934.128906 |
| normal | adam | 1.000000 | 5352.332031 |

Below are example traces of the cost vs epochs for xor and iris:

### XOR epochs
![epochs of xor](cost_xor.png?raw=true)

### Iris epochs
![epochs of iris](cost_iris.png?raw=true)
