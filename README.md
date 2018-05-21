# ComputerVision
Implementations of algorithms and operations used in Computer Vision

## Neural Style Transfer

To use the Neural Style transfer implementation:

```python
from NeuralStyleTransfer import implementNTS
```

initialize the neural style transfer implementation and train and generate an image
```python
train_func = implementNTS.init(num_iteration=1000)
generated_image = train_func()
```
the `init` function in the module implementNTS takes 3 arguments: <br/>
<br/>
num_iteration: number of iterations to train for<br/>
c_image: the path to the content image of size: (WIDTH=400, HEIGHT=300)<br/>
s_image: the path to the style image of size: (WIDTH=400, HEIGHT=300)<br/>


### Requirement
* Tensorflow
* numpy
* scipy

#### The code uses python3
