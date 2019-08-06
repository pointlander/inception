# Inception: compressible neural networks with faster convergence
A normal neural network layer has the form of sigmoid(A\*X + B). This experiment sets A = A<sub>1</sub>\*A<sub>2</sub> + A<sub>3</sub> and B = B<sub>1</sub>\*B<sub>2</sub> + B<sub>3</sub> so that the layer is now of the form sigmoid((A<sub>1</sub>\*A<sub>2</sub> + A<sub>3</sub>)\*X + B<sub>1</sub>\*B<sub>2</sub> + B<sub>3</sub>). This form results in faster convergence, and the terms (A<sub>1</sub>\*A<sub>2</sub> + A<sub>3</sub>) and (B<sub>1</sub>\*B<sub>2</sub> + B<sub>3</sub>) can be compressed down to matrices after training.

## Results
This idea has been tested on two problems: xor function and the iris dataset. In repeated experiments the xor function is learned ~10x faster and the iris dataset is learned ~2x faster. The faster learning rate does come at considerable computational cost. The increased rate is likely the result of an interplay between the optimizer and the network weights.

### XOR Experiments
| Mode      | Optimizer | Batch | Converged | Epochs      |
| --------- | --------- | ----- | --------- | ----------- |
| inception | static    | 1     | 0.898438  | 126.186957  |
| inception | static    | 4     | 0.941406  | 230.319502  |
| inception | momentum  | 1     | 0.902344  | 233.961039  |
| inception | adam      | 1     | 0.933594  | 260.815900  |
| inception | momentum  | 4     | 0.949219  | 271.798354  |
| inception | adam      | 4     | 0.937500  | 422.245833  |
| normal    | momentum  | 1     | 1.000000  | 1231.964844 |
| normal    | static    | 1     | 0.996094  | 1362.729412 |
| normal    | momentum  | 4     | 0.976562  | 3157.240000 |
| normal    | static    | 4     | 0.972656  | 3480.598394 |
| normal    | adam      | 4     | 0.855469  | 4882.771689 |
| normal    | adam      | 1     | 0.835938  | 7106.644860 |

### Iris Experiments
| Mode      | Optimizer | Batch | Converged | Epochs      |
| --------- | --------- | ----- | --------- | ----------- |
| inception | static    | 1     | 1.000000  | 146.875000  |
| inception | momentum  | 1     | 1.000000  | 155.332031  |
| inception | momentum  | 10    | 1.000000  | 206.394531  |
| inception | static    | 10    | 1.000000  | 213.019531  |
| normal    | momentum  | 1     | 1.000000  | 303.191406  |
| inception | adam      | 1     | 1.000000  | 315.460938  |
| normal    | static    | 1     | 1.000000  | 334.281250  |
| inception | adam      | 10    | 1.000000  | 557.566406  |
| normal    | adam      | 1     | 1.000000  | 1979.746094 |
| normal    | momentum  | 10    | 1.000000  | 3565.992188 |
| normal    | static    | 10    | 1.000000  | 3934.128906 |
| normal    | adam      | 10    | 1.000000  | 5352.332031 |

Below are example traces of the cost vs epochs for xor and iris:

### XOR epochs - batch size 4
![epochs of xor](cost_xor.png?raw=true)

### Iris epochs - batch size 10
![epochs of iris](cost_iris.png?raw=true)
