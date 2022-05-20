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
