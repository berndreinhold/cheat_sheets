_These notes are personal and do not abide to common scientific standards. Sentences and paragraphs may have been rewritten word by word without proper citation. (I memorize while (re)writing)_

# Neural Networks
## Notes from CS230 @Stanford (Youtube)
### lecture 1
Regression is harder than classification to train. ()

### lecture 2
- momentum

#### Loss function

activation function
Optimizer (Adam, ...)
Hyperparameters

image2vector: 3D-matrix (3rd dimension: RGB)
 
how could the input look like for diabetes data?

train on the BG and the difference between the predictions.

wx + b (w: weight, b; bias)

imbalance

one hot: softmax - assumes an object can occur only once
ReLU

labeling: 

fully connected vs. convolutional neural networks

"encoding"

Tips and Tricks:
- analyze a problem form a deep learning approach
- choose an architecture
- choose a loss and a training strategy

shipping a model: 
- architecture
- parameters

### heuristics
- gauge the complexity of a problem
- comparing to project in the past
- 10k images for a cat classifier
- outside/inside: just 10k
- difficult dawn, twilight, ...

- what resolution?
- the smaller the resolution the more efficient
- minimum resolution
- test with humans: print images: (64,64,3)

- shallow network, 
- loss function: how to choose loss function
- convex function (better to minimize)
- L1 (harder for classification, use it for regression problems)
- more resolution necessary (412, 412, 3)

- boost the dataset with files that are not well classified.

## Face Verification and Recognition
- define a 128-d vector as output of the neural network
- FaceNet
- Anchor, Positive, Negative
- define loss-function: L = |eps(A)-eps(P)|^2 - |eps(A)-eps(N)|^2
    - add an alpha term
    - minimize max(L, 0)
- minimize that loss-function
- what do these vectors then represent?
- 

## Neural Style Transfer
- we are not learning parameters by minimizin L. We are learning an image!
- start with white noise
- go back to the image: update the pixels
- art image: just to run once for the style
- edges look like the content 
- 2000 iterations for the pixels

## Trigger word detection
- 10 sec audio speech, detect the word "activate"
- resolution of an image
- RNN: 1 right after the keyword
- many 0, few 1: many hacks in papers.
- RNN: go through sequentially
- logistic regression

- positive word
- negative words
- background noise

### Tips
- talk to an expert
- build a network of people
- error analysis

# Glossary

# LSTM
Brandon Rohrer, 2017: Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)

- Forgetting/memory (with gate)
- ignoring (with gate)
- selection (with gate)

squashing functions: sigmoid (0,1) or tanh (-1,1)

children book: Jane saw Spot

RNN: can look back a few time steps
LSTM: can look back much longer

Text translation: phrase to phrase or sentence to sentence translation
Robotics: agent taking in information, decision, action (LSTM looks )

# Example Networks and Architectures
## AlexNet
- Tremendous improvement on ImageNet competition in 2012


# Artificial General Intelligence (AGI)
- https://en.wikipedia.org/wiki/Artificial_general_intelligence
- AGI is the ability of an intelligent agent to understand or learn any intellectual task that a human being can.
- strong AI: machine showing traits of consciousness and sentience (feelings and sensation)

## Tests
- Turing Test: conversation between test human and machine and human, the test human has to decide whether a machine is involved
- The Coffee Test (Wozniak): a machine has to enter an average American home and figure out how to make coffee: find the coffee machine, the water, a mug, brew coffee (and drink it)
- robot college student test (Goertzel): a machine applies to colleague, participates in the coarses and gets a degree
- employment test (Nilsson): a machine performs an economically important job at least as well as humans in the same job

## Trivia
- Ben Goertzel: "achieving complex goals in complex environments"
- Animal Tests: Animal Olympiad for AI
- AI hard

# Deep Learning
by Ian Goodfellow, Yoshua Bengio, Aaron Courville

### 5.2 Capacity, Overfitting and Underfitting
"What separates machine learning from optimization is that we want the generalization error, also called the test error, to be low as well. The generalization error is defined as the expected value of the error" on previously unseen input.

Capacity: set of functions allowed in the algorithms hypothesis psace

Error vs. Capacity
Underfitting (low capacity): test and training error are both big
Overfitting (high capacity): training error decreases, but test error increases and a generalization gap appears that is widening with further increasing capacity.

#### 5.2.1 No Free Lunch Theorem
We must train on specific tasks, relevant in real world settings, because a general learning algorithm won't generalize well. (Note to self: Hmmm, that's too short a summary and probably missing the point)
#### 5.2.2 Regularization
The focus on a specific task opens possibilities to tune the learning algorithm with additional constraints that help generalization.
One can give preferences to certain solutions in the space of hypothesis and parameters.

Regularization examples:
- weight decay: a term in the loss function preferring small weights
- many others, see chapter 7 :)

"Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error. Regularization is one of the central concerns of the field of machine learning, rivaled in its importance only by optimization."

#### 5.7.2 Support Vector Machines
kernel trick :)
The kernel-based function is exactly equivalent to preprocessing the data by applying $phi(x)$ to all inputs, then learning a linear model in the new transformed space.
## 7 Regularization for Deep Learning
Strategies designed to reduce the test error even if at the cost of a higher training error are called regularization.
Many kinds of regularizations have been developed:
- additional components in the loss (aka cost) function, e.g. parameter norm penalties
- these additional components can add prior knowledge or preferences to the loss function
- ensemble methods are also a kind of regularization
- in the context of deep learning there is a focus on regularizing estimators. Regularization of estimators works by trading increased bias for reduced variance

### 6.4 Architecture Design
- depth of the network, 
- width of the layer
- "Deeper networks are often abel to user far fewer units per layer and far fewer parameters, as well as frequently generalizing to the test set, but they also tend to be harder to optimize. The ideal network architecture for a task must be found via experimentation guided by monitoring the validation set error."
#### 6.4.2 Other architectural considerations
- there are much more variants in neural network architectures beyond depth of the network and width of the layer.
- feedforward networks:
    - convolutional networks
    - recurrent neural networks for sequence processing
- layers do not need to be connected in a chain
- how is layer $l$ connected to layer $l + 1$
- fully connected: every output of layer $l$ with input of layer $l + 1$
- specialized networks have layers where only subsets of output nodes are connected to input nodes of layer $l + 1$
- convolutional networks are highly specialized for computer vision problems.

### 6.5 Back-Propagation and Other Differentiation Algorithms
- "When we use a feedforward neural network to accept an input $x$ and produce an output $y^{hat}$" information flows forward through the network. The input $x$ provides the initial information that then propagates up to the hidden units at each layer and finally produces $y^{hat}$. This is called *forward propagation*. During training, forward propagation can continue onward until it produces a scalar cost $J(\theta)$."
- "The back-propagation algorithm (Rumelhart et al., 1986a), often simply called backprop, allows the information from the cost to then flow backward through the network in order to compute the gradient."
- back-propagation: calculate the gradient
- stochastic gradient descent: used to perform learning using this gradient
- backprop not specific to multilayer neural networks, "in principle can compute derivatives of any function (for some functions, the correct response is to report that the derivative of the function is undefined)"
- "The idea of computing derivatives by propagating information through a network is very general and can be used to compute values such as the Jacobian of a function f with multiple outputs. We restrict our description here to the most commonly used case, where f has a single output."

#### 6.5.1 Computational Graphs
- So far we have discussed neural networks with a relatively informal graph language.
- formalize the computational graph language
- each node in the graph indicates a variable. "The variable may be a scalar, vector, matrix, tensor, or even a variable of another type."
- "An *operation* is a simple function of one or more variables. Our graph language is accompanied by a set of allowable operations. Functions more complicated than the operations in this set may be described by composing many operations together. Without loss of generality we define an operation to return only a single output variable. (...)
If a variable y is computed by applying an operation to a variable x, then we draw a directed edge from x to y. We sometimes annotate the output node with the name of the operation applied."

#### 6.5.2 Chain Rule of Calculus
- "The chain rule of calculus (not to be confused with the chain rule of probability) is used to compute the derivatives of functions formed by composing other ufcntions, whose derivates are known. Back-propagation is an algorithm that computes the chain rule, with a specific order of operations that is highly efficient."
- $\frac{dz}{dx} = \frac{dz}{dy}\cdot\frac{dy}{dx}$ with $y = g(x)$ and $z = f(g(x))$
- $\frac{\delta y}{\delta x}$ is the Jacobian matrix of $g$.
- "From this we see that the gradient of a variable $x$ can be obtained by multiplying a Jacobian matrix $\frac{\delta y}{\delta x}$ with a gradient $\Delta_y^z$. The back-propagation algorithm consist of performing such a Jacobian-gradient product for each operation in the graph."

#### 6.5.3 Recursively Applying the Chain Rule to Obtain Backprop
- "Using the chain rule, it is straightforward to write down an algebraic expression for the gradient of a scalar with respect to any node in the computational graph that produced that scalar. Actually evaluating that expression in a computer, however, introduces some extra considerations."
- many operations need to be done several times: trade-off between calculating them anew each time resulting in larger run time vs. a larger memory footprint

### 6.6 Historical Notes
- "Feedforward networks can be seen as efficient nonlinear function approximators based on using gradient descent to minimize the error in a function approximation. From this point of view, the modern feedforward network is the culmination of centuries of progress on the general function approximation task."
- "gradient descent was not introduced as a technique for iteratively approximating the solution to optimization problems until the nineteenth century (Cauchy, 1847).
- flaws of the linear model family: "inability to learn the XOR function, which lead to a backlash against the entire neural network approach." 
- "connectionism", "distributed representation"
- "Following the success of back-propagation, neural network research gained popularity and reached a peak in the early 1990s. Afterwards, other machine learning techiques became more popular until the modern deep learning renaissance that began in 2006."
- core ideas behind feedforward networks are the same, also the back-propagation algorithm and gradient descent. 
- three factors for improvements:
    - "larger datasets have reduced the degree to which statistical generalization is a challenge for neural networks"
    - "Second, neural networks have become much larger, because of more powerful computers and better software infrastructure."
    - "a small number of algorithmic changes have also improved the performance of neural networks noticeably."

#### Algorithmic changes
- cross entropy (log loss) function rather than mean squared error as loss function for networks with sigmoid outputs. (previous: saturation and slow learning)
- ReLU instead of sigmoid (which performs better for small networks): AlexNet used ReLU, LeNet used sigmoid. False Assumption against ReLU: Activation functions with nondifferentiable points must be avoided.
- Biology as inspiration: Neurons: "(1) for some inputs, biological neurons are completely inactive. (2) For some inputs, a biological neuron's output is proportional to its input. (3) Most of the time, biological neurons operate in the regime where they are inactive (i.e. they should have sparse activations)."

From about 2006 to 2012 it was widely believed that feedforward networks would not perform well unless they were assisted by other models, such as probabilistic models.


### 7.1 Parameter Norm Penalties
- for neural networks a norm penalty is used that addresses only the weights of the affine transformations at each layer, but not their biases.
- experience shows that regularizing the biases can introduce underfitting

### 7.3 Regularization and Under-Constrained Problems
many linear models in ML rely on inverting $X^T\cdot X$, which is only possible, if X is not singular.

### 7.4 Dataset Augmentation
- Train a model on more data by generating fake data. For some ML problems it is reasonably easy to produce fake data.
- Easy example: classifiers
- not so easy: difficult for problems trying to perform density estimations. Creating new fake data relies on knowing the underlying density estimation.
- easy: shifts or rotations of images used for object detection
- hard: out-of-plane rotations
- works well for speech recognition tasks
- injecting noise

But: neural networks turn out to be not very robust to noise. ($\rightarrow$ Adversary attacks)

- dropout: a powerful regularization strategy can be seen as a process of constructing new inputs by multiplying by noise

### 7.5 Noise Robustness
- adding noise to the weights
- noise also added to deeper layers (different layers of abstraction)

### 7.9 Parameter Tying and Parameter Sharing
- link two weight distribution via a regularization term in the loss function or
- enforce sharing of parameters
- much reduced memory footprint, e.g. in convolutional neural networks

#### 7.9.1 Convolutional Neural Networks
"By far the most popular and extensive use of parameter sharing occurs in convolutional neural networks (CNNs) applied to computer vision.
Natural images have many statistical properties that are invariant to translation.
"
invariance to translation
see chapter 9 for a deeper discussion

### 7.11 Bagging and Other Ensemble Methods
Bagging: bootstrap aggregating

benchmark comparisons in scientific papers are done on single models.
In real life averages of dozens of models outperform single models.

### 7.12 Dropout
"Dropout provides a computationally inexpensive but powerful method of regularizing a broad family of models."
"It trains an ensemble of all subnetworks that can be constructed by removing nonoutput units from an underlying base network."