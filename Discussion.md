# Discussion for Deep Learning 

### Date : 2018-11-06

#### __1. Data Preprocessing__ <br>
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

Reference : [Image preprocessing in Deep Learning](https://stackoverflow.com/questions/41428868/image-preprocessing-in-deep-learning)
<br>
<br>

#### __2. Early Stopping__ <br>
Early stopping is widely used to stop learning while monitoring the performance curve. Today in Deep learning Course, the professor said early stopping is unnoticeable which means there is no change in learning dynamics. If I had done early stopping with basic cost, whether how I add terms in cost function I will do early stopping anyway.<br>
I was curious what should be the index to do early stopping. When dev-set cost is minimum? Or when dev-set accuracy is maximum? So I searched through internet, and I found one answer. Which says it is case-by-case. The ultimate goal of the model will determine what to monitor. 
<br>
Reference : [Early stopping on validation loss or on accuracy?](https://datascience.stackexchange.com/questions/37186/early-stopping-on-validation-loss-or-on-accuracy) <br>

Of course, early stopping is not a really good option due to breaking orthogornalization of parameters(it breaks down more than one parameter simulataneously). But for homework, I can't just wait and monitor the curve even if validation cost is going on! So I think it would be useful to save time when we use early stopping well.
<br>

**18/12/01** <br>
[Capturing keyboardinterrupt](https://stackoverflow.com/questions/4205317/capture-keyboardinterrupt-in-python-without-try-except)
[Early stop using Keras](https://chrisalbon.com/deep_learning/keras/neural_network_early_stopping/)

<br>

**3. Batch size** <br>
[Batch Size in Deep Learning](https://blog.lunit.io/2018/08/03/batch-size-in-deep-learning/)
<br>
[ON LARGE-BATCH TRAINING FOR DEEP LEARNING: GENERALIZATION GAP AND SHARP MINIMA](https://openreview.net/pdf?id=H1oyRlYgg)


#### **4. Fluctuation in Performance Curve** <br>
While I was training 2 layers neural netowrk, sometime there were lots of peaks in training&dev-set cost.
There were some points with very large cost peaking out. 
  - Other reference : [37 Reasons why your neural netowork is not working](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607] <br>
  
  
**5. Pyplot** <br>
[two_scales graph(twin)](https://matplotlib.org/examples/api/two_scales.html) <br>


#### **6. Save & Restore in Tensorflow** <br>


- References<br>
[Link1](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/) <br>
[TensorFlow 모델을 저장하고 불러오기](http://goodtogreate.tistory.com/entry/Saving-and-Restoring) <br>
[텐서플로우(TensorFlow)에서 Tf.Train.Saver API를 이용해서 모델과 파라미터를 저장(Save)하고 불러오기(Restore)](http://solarisailab.com/archives/2524) <br>
[Save and Restore](https://www.tensorflow.org/guide/saved_model) <br>
(http://jaynewho.com/post/8) <br>
