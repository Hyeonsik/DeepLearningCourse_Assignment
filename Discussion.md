# Discussion for Deep Learning 

### Date : 2018-11-06

1. Data Preprocessing <br>

Basically it is better to make data zero-centered with standard deviation 1 (normalized). <br>
I quoted explantion from Standford CS231n 2016 lectures :
> Normalization refers to normalizing the data dimensions so that they are of approximately the same scale. For Image data There are two common ways of achieving this normalization. One is to divide each dimension by its standard deviation, once it has been zero-centered:
(X /= np.std(X, axis = 0)). Another form of this preprocessing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively. It only makes sense to apply this preprocessing if you have a reason to believe that different input features have different scales (or units), but they should be of approximately equal importance to the learning algorithm. In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step.
<br>

Here is how I applied to my homework. (Artificial Intelligence System, 2018 Fall, Homework4)
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    x = (x - tf.reduce_mean(x, axis = 0, keepdims = True))
    y = (y - tf.reduce_mean(y, axis = 0, keepdims = True))

Because image data is already boundaried between 0 to 1(in this case), I only moved data to make it zero-centered. Normalizing data would take some computations and time, I just only did zero-centered. And it works better than before on fitting 3-layers neural networks. I think it's because input data has lots of noises with few numbers of data.

Reference : [Image preprocessing in Deep Learning]("https://stackoverflow.com/questions/41428868/image-preprocessing-in-deep-learning")
<br>

2. 
