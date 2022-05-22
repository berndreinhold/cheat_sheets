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

"Regularization is any modification we make to a learning algorithm that is intended to reduce its genearlization error but not its training error. Regularization is one of the central concerns of the field of machine learning, rivaled in its importance only by optimization."

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

But: neural networks turn out to be not very robust to noise. ($\righarrow$ Adversary attacks)

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