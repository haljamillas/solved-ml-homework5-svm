Download Link: https://assignmentchef.com/product/solved-ml-homework5-svm
<br>
<strong> SVM. </strong>In this exercise, we will explore different kernels for SVM and the relation between the parameters of the optimization problem. We will use an existing implementation of SVM: the SVC class from sklearn.svm. This class solves the soft-margin SVM problem. You can assume that the data we will use is separable by a linear separator (i.e. that we could formalize this problem as a hard-margin SVM). In the file skeleton svm.py you will find two implemented methods:

<ul>

 <li>get points – returns training and validation sets of points in 2D, and their labels.</li>

 <li>create plot – receives a set of points, their labels and a trained SVM model, and creates a plot of the points and the separating line of the model. The plot is created in the background using matplotlib. To show it simply run show() after calling this method.</li>

</ul>

In the following questions, you will be asked to implement the other methods in this file, while using get points and create plot that were implemented for you.

<ul>

 <li><strong> </strong>Implement the method train three kernels that uses the training data to train 3 kernel SVM models – linear, quadratic and RBF. For all models, set the penalty constant <em>C </em>to 1000.

  <ul>

   <li>How are the 3 separating lines different from each other? You may support your answer with plots of the decision boundary.</li>

   <li>How many support vectors are there in each case?</li>

  </ul></li>

 <li><strong> </strong>Implement the method linear accuracy per C that trains a linear SVM model on the training set, while using cross-validation on the validation set to select the best penalty constant from <em>C </em>= 10<sup>−5</sup><em>,</em>10<sup>−4</sup><em>,…,</em>10<sup>5</sup>. Plot the accuracy of the resulting model on the training set and on the validation set, as a function of <em>C</em>.

  <ul>

   <li>What is the best <em>C </em>you found?</li>

   <li>Explain the behavior of error as a function of <em>C</em>. Support your answer with plots of the decision boundary.</li>

  </ul></li>

</ul>

(c) <strong> </strong>Implement the method rbf accuracy per gamma that trains an RBF SVM model on the training set, while using cross-validation on the validation set to select the best coefficient <em>γ </em>= 1<em>/σ</em><sup>2</sup>. Start your search on the log scale, e.g., perform a grid search <em>γ </em>= 10<sup>−5</sup><em>,</em>10<sup>−4</sup><em>,…,</em>10<sup>5</sup>, and increase the resolution until you are satisfied. Use <em>C </em>= 10 as the penalty constant for this section. Plot the accuracy of the resulting seperating line on the training set and on the validation set, as a function of <em>γ</em>.

<ul>

 <li>What is the best <em>γ </em>you found?</li>

 <li>How does <em>γ </em>affect the decision rule? Support your answer with plots of the decision boundary.</li>

</ul>

<ol start="2">

 <li><strong> Back-Propagation. </strong>In this exercise we will implement the back-propagation algorithm for training a neural network. We will work with the MNIST data set that consists of 60000 28×28 gray scale images with values of 0 to 1 in each pixel (0 – white, 1 – black). The optimization problem we consider is of a neural network with sigmoid activations and the cross entropy loss. Namely, let <strong>x </strong>∈ R<em><sup>d </sup></em>be the input to the network (in our case <em>d </em>= 784) and denote <strong>a</strong><sub>0 </sub>= <strong>x </strong>and <em>n</em><sub>0 </sub>= 784. Then for 0 ≤ <em>t </em>≤ <em>L </em>− 2, define</li>

</ol>

<em>z</em><em>t</em>+1 = <em>W</em><em>t</em>+1<strong>a</strong><em>t </em>+ <strong>b</strong><em>t</em>+1 <strong>a</strong><em>t</em>+1 = <em>h</em>(<strong>z</strong><em>t</em>+1) ∈ R<em>n</em><em>t</em>+1

and

<em>z<sub>L </sub></em>= <em>W<sub>L</sub></em><strong>a</strong><em><sub>L</sub></em>−1 + <strong>b</strong><em><sub>L</sub></em>

<em>e</em><strong>z</strong><em>L</em>

<strong>a</strong><em>L </em>=  <em>L</em>

P<em>i e</em><strong>z</strong><em><sub>i</sub></em>

where <em>h </em>is the sigmoid function applied element-wise on a vector (recall the sigmoid function ) and <em>W<sub>t</sub></em><sub>+1 </sub>∈ R<em><sup>k</sup></em><em><sup>t</sup></em><sup>+1×<em>k</em></sup><em><sup>t</sup></em>, <strong>b</strong><em><sub>t</sub></em><sub>+1 </sub>∈ R<em><sup>k</sup></em><em><sup>t</sup></em><sup>+1 </sup>(<em>k<sub>t </sub></em>is the number of neurons in layer <em>t</em>). Denote by W the set of all parameters of the network. Then the output of the network (after the softmax) on an input <strong>x </strong>is given by <strong>a</strong><em><sub>L</sub></em>(<strong>x</strong>;W) ∈ R<sup>10 </sup>(<em>n<sub>L </sub></em>= 10).

Assume we have an MNIST training data set  where <strong>x</strong><em><sub>i </sub></em>∈ R<sup>784 </sup>is the 28×28 image given in vectorized form and <strong>y</strong><em><sub>i </sub></em>∈ R<sup>10 </sup>is a one-hot label, e.g., (0<em>,</em>0<em>,</em>1<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0) is the label for an image containing the digit 2. Define the log-loss on a single example (<strong>x</strong><em>,</em><strong>y</strong>), <em>`</em><sub>(<strong>x</strong><em>,</em><strong>y</strong>)</sub>(<em>W</em>) = −<strong>y </strong>· log<strong>a</strong><em><sub>L</sub></em>(<strong>x</strong>;W) where the logarithm is applied element-wise on the vector <strong>a</strong><em><sub>L</sub></em>(<strong>x</strong>;W). The loss we want to minimize is then

The code for this exercise is given in the backprop.zip file in moodle. The code consists of the following:

<ul>

 <li>py: Loads the MNIST data.</li>

 <li>py: Code for creating and training a neural network.</li>

 <li>py: Example of loading data, training a neural network and evaluating on the testset.</li>

 <li>pkl.gz: MNIST data set.</li>

</ul>

The code in network.py contains the functionality of the training procedure except the code for back-propagation which is missing.

Here is an example of training a one-hidden layer neural network with 40 hidden neurons on a randomly chosen training set of size 10000. The evaluation is performed on a randomly chosen test set of size 5000. It trains for 30 epochs with mini-batch size 10 and constant learning rate 0.1.

&gt;&gt;&gt; training_data, test_data = data.load(train_size=10000,test_size=5000)

&gt;&gt;&gt; net = network.Network([784, 40, 10])

&gt;&gt;&gt; net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

<ul>

 <li><strong>(6 points) </strong>Implement the back-propagation algorithm in the <em>backprop </em>function in the <em>Network </em> The function receives as input a 784 dimensional vector <strong>x </strong>and a one-hot vector <strong>y</strong>. The function should return a tuple (<em>db,dw</em>) such that <em>db </em>contains a list of derivatives of <em>`</em><sub>(<strong>x</strong><em>,</em><strong>y</strong>) </sub>with respect to the biases and <em>dw </em>contains a list of derivatives with respect to the weights. The element <em>dw</em>[<em>i</em>] (starting from 0) should contain the matrix and <em>db</em>[<em>i</em>] should contain the vector .</li>

</ul>

In order to check your code you can use the following approximation of a one-dimensional derivative  for small . You can use this to check that your partial derivative calculations are correct. You can use the <em>loss </em>function in the <em>Network </em>class to calculate <em>`</em><sub>(<strong>x</strong><em>,</em><strong>y</strong>)</sub>(W).

<ul>

 <li><strong> </strong>Train a one-hidden layer neural network as in the example given above (e.g., training set of size 10000, one hidden layer of size 40). Plot the <em>training </em>accuracy, <em>training </em>loss (<em>`</em>(W)) and <em>test </em>accuracy across epochs (3 different plots). For the test accuracy you can use the <em>one label accuracy </em>function, for the training accuracy use the <em>one hot accuracy </em>function and for the training loss you can use the <em>loss </em> All functions are in the <em>Network </em>class. The test accuracy in the final epoch should be above 80%.</li>

 <li><strong> </strong>Now train the network on the whole training set and test on the whole test set:</li>

</ul>

&gt;&gt;&gt; training_data, test_data = data.load(train_size=50000,test_size=10000)

&gt;&gt;&gt; net = network.Network([784, 40, 10])

&gt;&gt;&gt; net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

Do <strong>not </strong>calculate the training accuracy and training loss as in the previous section (this is time consuming). What is the test accuracy in the final epoch (should be above 90%)?

<ul>

 <li><strong> </strong>In this section we will train a deep network with 4 hidden layers using gradient descent (i.e., mini-batch size=training set size) and observe the vanishing gradient phenomenon. This occurs when the gradients of the layers closer to the input have lower norms than gradients of the hidden layers that are closer to the output. In other words, the early hidden layers are learned more slowly than later layers. In each epoch calculate the gradient euclidean norms for 0 ≤ <em>i </em>≤ 4. These are the gradients with respect to the whole training set. Do not forget to divide by the size of the training set as in <em>`</em>(W). Train a 4 hidden layer neural network with 30 hidden neurons in each layer as follows:</li>

</ul>

&gt;&gt;&gt; training_data, test_data = data.load(train_size=10000,test_size=5000)

&gt;&gt;&gt; net = network.Network([784, 30, 30, 30, 30, 10])

&gt;&gt;&gt; net.SGD(training_data, epochs=30, mini_batch_size=10000, learning_rate=0.1, test_data=test_data)

Plot the values  for 0 ≤ <em>i </em>≤ 4 across epochs (one plot). Using the expression of the gradients in the backpropagation algorithm and the derivative of the sigmoid, give a possible explanation for this phenomenon.

<ol start="3">

 <li><strong> Multilayer Perceptron. </strong>In this exercise you will implement a multilayer perceptron (MLP) network, and explore how training is affected by the network architecture. We will work with Keras, a popular high-level deep learning framework.</li>

</ol>

To start, download the files skeleton mlp mnist.py and mlp main.py from Moodle. The file skeleton mlp mnist.py is a non-complete implementation of the MLP network and its training. The file mlp main.py is the main program, that builds the network defined in the other file and train and evaluate it on the MNIST dataset. It also stores useful graphs, with model accuracy and loss value as the y-axis and the number of epochs<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> or model complexity as the x-axis. See the help menu or the code for more details.

You should add code only at the designated places, as specified in the instructions. In your submission, make sure to include your code, written solution and output graphs.

<ul>

 <li><strong> </strong>MLP is a basic neural network architecture that consists of at least three layers of nodes. Except for the input layer, on each layer a nonlinear activation function is applied. Implement the class method build model no skip in skeleton mlp py. This method should build a network according to a given set of parameters. Specifically, it should build the following layers:

  <ul>

   <li><em>k </em>fully-connected layers of dimensions {<em>d</em><sub>1</sub><em>,…,d<sub>k</sub></em>}. For these layers, use the ReLU<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>activation function.</li>

   <li>An output layer of size that equals to the number of classes. For this layer, we would like to use softmax<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> as an activation layer, in order to get a probability distribution over the classes.</li>

  </ul></li>

</ul>

In your implementation, create a Sequential model and use the Dense API<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a>. Further details are provided in the code.

Notice how we compile the model with SGD as the optimizer. This is a great benefit of deep learning frameworks, that ship with built-in automatic differentiation and various optimization methods.

<ul>

 <li>Perform a series of 5 experiments of increasing network complexity. Run the program in series mode, and specify an increasing number of hidden layers and hidden units, such that in each experiment the network has more parameters than the one before. For now, we will use relatively shallow networks, with no more than 6 layers and no more than 512 hidden units. Please keep all other parameters to default (batch size, number of epochs, etc.).</li>

</ul>

Examining the output graph, how does adding more parameters affect the training?

<ul>

 <li><strong> </strong>Now we will go deeper. Run the program in single mode, with 80 epochs and 60 hidden layers of 16 units each. Examining the two output graphs, how does the training process looks like? Could you give a possible explanation?</li>

 <li><strong> </strong>A naive implementation of deeper networks does not always work. A common practice for these cases is skip connections, which are connections that skip not one but multiple layers<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a>. Concretely, if <strong>x </strong>is the value of a hidden layer before applying non-linearity <em>σ</em>, and <strong>z </strong>= F(<strong>x</strong>) is the value of another layer down the network, then a</li>

</ul>

skip connection between the two layers is created by replacing <em>σ</em>(<strong>z</strong>) with <em>σ</em>(<strong>x </strong>+ <strong>z</strong>) = <em>σ</em>(<strong>x </strong>+ F(<strong>x</strong>)) (see Figure 1 for an illustration).

Figure 1: Skip connection between two hidden layers, <strong>x </strong>and <strong>z </strong>are the layer values before applying a non-linearity function <em>σ</em>.

Implement the class method build model skip in skeleton mlp mnist.py, that creates skip connections between every <em>n </em>layers. The parameter <em>n </em>is determined by the class parameter skips, which is set according to the program argument.

In this question, use the Model functional API<a href="#_ftn6" name="_ftnref6"><sup>[6]</sup></a> and not the Sequential method.

<strong> </strong>Run the same experiment from the previous question (c), but now with skip connections between every 5 layers (use the skips parameter). Compare the output graphs to the results in the previous question

<a href="#_ftnref1" name="_ftn1">[1]</a> The number of passes on the entire training data.

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">https://en.wikipedia.org/wiki/Rectifier_(neural_networks)</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> <a href="https://en.wikipedia.org/wiki/Softmax_function">https://en.wikipedia.org/wiki/Softmax_function</a>

<a href="#_ftnref4" name="_ftn4">[4]</a> <a href="https://keras.io/layers/core/#dense">https://keras.io/layers/core/#dense</a>

<a href="#_ftnref5" name="_ftn5">[5]</a> You can read more about skip connections in this paper: <a href="https://arxiv.org/abs/1512.03385">https://arxiv.org/abs/1512.03385</a>

<a href="#_ftnref6" name="_ftn6">[6]</a> <a href="https://keras.io/models/model/">https://keras.io/models/model/</a>