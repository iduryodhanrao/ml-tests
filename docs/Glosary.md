# 📖 Deep Learning Glossary

A comprehensive glossary of terms used by data scientists and machine learning engineers.

---
### A

**Activation Function**
A function applied to the output of a neuron that determines its final output. It introduces non-linearity into the model, allowing it to learn complex patterns. Examples include ReLU, Sigmoid, and Softmax.

**Adam (Adaptive Moment Estimation)**
A popular optimization algorithm that adapts the learning rate for each parameter, combining the advantages of both AdaGrad and RMSProp optimizers.

**Autoencoder**
A type of neural network used for unsupervised learning, typically for dimensionality reduction or feature learning. It consists of an encoder that compresses the data and a decoder that reconstructs it.

**Autoregressive Model**
A generative model that creates sequences one element at a time, where each new element is conditioned on the previous ones (e.g., GPT, WaveNet).

---
### B

**Backpropagation**
The core algorithm for training neural networks. It calculates the gradient of the loss function with respect to the network's weights, allowing the model to update the weights via gradient descent.

**Batch Normalization**
A technique used to stabilize and speed up the training of deep neural networks by normalizing the inputs to each layer.

**BERT (Bidirectional Encoder Representations from Transformers)**
A powerful language model based on the Transformer architecture that learns deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.

**Bias**
A parameter in a neural network that allows it to shift the activation function to the left or right, adding flexibility to the model. In machine learning more broadly, bias refers to a model's systematic error or tendency to make consistent errors on unseen data.

**BLEU Score (Bilingual Evaluation Understudy)**
A metric for evaluating the quality of machine-translated text by comparing it to high-quality human translations.

---
### C

**Convolutional Neural Network (CNN)**
A class of deep neural networks, most commonly applied to analyzing visual imagery. They use convolutional layers to filter inputs for useful information.

**Cross-Entropy**
A widely used loss function for classification tasks that measures the difference between two probability distributions: the predicted output and the true labels.

**Cross-Validation**
A resampling procedure used to evaluate machine learning models on a limited data sample, such as k-fold cross-validation.

---
### D

**Data Augmentation**
A technique to increase the diversity of a training dataset by applying random transformations (e.g., rotation, cropping, flipping) to the existing data.

**Dense Layer (Fully Connected Layer)**
A layer in a neural network where every neuron is connected to every neuron in the previous layer.

**Discriminator**
In a GAN, the network that tries to distinguish between real data and the fake data created by the generator.

**Dropout**
A regularization technique where randomly selected neurons are ignored during training, preventing the model from becoming too reliant on any single neuron and thus reducing overfitting.

**Dynamic Programming**
In reinforcement learning, a collection of algorithms that can compute optimal policies given a perfect model of the environment.

---
### E

**Embeddings**
Dense, low-dimensional vector representations of high-dimensional or categorical data, such as words (Word2Vec) or items.

**Epoch**
One complete pass through the entire training dataset.

**Exploding Gradients**
A problem during training where the gradients grow exponentially large, leading to unstable updates and preventing the model from converging.

---
### F

**F1-Score**
A metric for a model's accuracy on a classification task, calculated as the harmonic mean of precision and recall.

**Faster R-CNN**
A popular two-stage object detection model that integrates region proposal generation into the network for end-to-end training.

**Feature Map**
The output of one filter applied to the previous layer in a CNN. It represents the presence of a specific feature in the input.

**Feedforward Neural Network**
The simplest type of artificial neural network, where connections between nodes do not form a cycle.

**Fine-Tuning**
The process of taking a pre-trained model and re-training it on a new, often smaller, dataset for a specific task.

**Focal Loss**
A loss function designed to address class imbalance in one-stage object detectors by down-weighting the loss assigned to well-classified examples.

---
### G

**GAN (Generative Adversarial Network)**
A class of generative models where two networks, a generator and a discriminator, are trained in a competitive game to create realistic synthetic data.

**Generator**
In a GAN, the network that learns to create fake data that is realistic enough to fool the discriminator.

**GloVe (Global Vectors for Word Representation)**
A popular algorithm for learning word embeddings by analyzing a global word-word co-occurrence matrix.

**GPT (Generative Pre-trained Transformer)**
A family of powerful language models based on the Transformer architecture, capable of generating human-like text.

**Gradient Descent**
An iterative optimization algorithm used to find the minimum of a function (the loss function) by repeatedly moving in the direction of the steepest descent.

---
### H

**Hidden Layer**
Any layer in a neural network between the input and output layers.

**Hyperparameter**
A configuration setting that is external to the model and whose value cannot be estimated from data, such as the learning rate, number of epochs, or batch size.

---
### I

**Image Classification**
The task of assigning a single label to an entire image (e.g., "cat" or "dog").

**Image Segmentation**
The task of partitioning a digital image into multiple segments (sets of pixels), often to locate objects and their boundaries.

**Inception Module**
A building block for CNNs (used in GoogLeNet) that performs convolutions with multiple different filter sizes in parallel.

---
### K

**Kernel (or Filter)**
A small matrix used in a convolutional layer to perform a convolution operation, detecting specific features like edges or textures.

**K-Fold Cross-Validation**
A specific type of cross-validation where the data is split into 'k' folds, and the model is trained k times, each time using a different fold as the test set.

---
### L

**Latent Space**
A low-dimensional, compressed representation of data, often learned by autoencoders or GANs.

**Layer**
A collection of neurons that perform a specific transformation on their input data.

**Learning Rate**
A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.

**LeNet-5**
One of the earliest successful CNNs, designed for handwritten digit recognition.

**Long Short-Term Memory (LSTM)**
An advanced type of RNN that uses gates to regulate the flow of information, allowing it to learn long-range dependencies and avoid the vanishing gradient problem.

**Loss Function (or Cost Function)**
A function that measures the difference between the model's predictions and the actual ground truth labels. The goal of training is to minimize this function.

---
### M

**Mask R-CNN**
An object detection model that extends Faster R-CNN to also predict a pixel-level segmentation mask for each object.

**MobileNet**
A family of efficient CNNs designed for mobile and embedded vision applications.

**Mode Collapse**
A common failure mode in GAN training where the generator produces a very limited variety of samples.

**Model**
The output of a machine learning algorithm run on data; in deep learning, it refers to the network architecture and its learned weights.

**Monte Carlo Methods**
In reinforcement learning, model-free methods that learn directly from complete episodes of experience.

---
### N

**Named Entity Recognition (NER)**
An NLP task that involves identifying and classifying named entities (like persons, organizations, and locations) in text.

**Normalization**
The process of scaling numerical data to a standard range (e.g., 0 to 1) to improve the performance and stability of a model.

**NLP (Natural Language Processing)**
A field of AI focused on enabling computers to understand, interpret, and generate human language.

---
### O

**Object Detection**
The task of identifying and locating objects within an image, typically by drawing a bounding box around them.

**One-Hot Encoding**
A process of converting categorical variables into a binary vector representation where one bit is "hot" (1) and all others are 0.

**Optimizer**
An algorithm used to change the attributes of the neural network, such as its weights and learning rate, to reduce the losses (e.g., Adam, SGD, RMSprop).

**Overfitting**
A problem where a model learns the training data too well, including its noise and random fluctuations, and fails to generalize to new, unseen data.

---
### P

**Padding**
Adding extra pixels (usually zeros) around the border of an image before a convolution to control the output size.

**Parameter**
A variable internal to the model that is learned from the data, such as the weights and biases of a neural network.

**Pooling Layer**
A layer in a CNN that reduces the spatial dimensions of the feature maps, making the model more computationally efficient and robust to variations in feature position.

**Precision**
A classification metric that measures the proportion of true positive predictions among all positive predictions.

**Perplexity**
A metric used to evaluate language models, measuring how well a probability model predicts a sample. Lower is better.

**Pre-training**
The process of training a model on a large, general dataset before fine-tuning it on a specific task.

---
### Q

**Q-learning**
A model-free, off-policy reinforcement learning algorithm that learns a policy telling an agent what action to take under what circumstances.

---
### R

**Recall (or Sensitivity)**
A classification metric that measures the proportion of actual positives that were correctly identified.

**Recurrent Neural Network (RNN)**
A type of neural network designed for sequential data, where connections between nodes form a directed graph along a temporal sequence.

**ReLU (Rectified Linear Unit)**
A popular activation function that outputs the input directly if it is positive, and zero otherwise.

**Reinforcement Learning (RL)**
An area of machine learning where an agent learns to make decisions by performing actions in an environment to maximize a cumulative reward.

**ResNet (Residual Network)**
A groundbreaking CNN architecture that introduced "skip connections" to allow for the training of extremely deep networks.

**R-squared ($R^2$)**
A statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.

---
### S

**SARSA (State-Action-Reward-State-Action)**
An on-policy reinforcement learning algorithm that learns a policy based on the actions actually taken by the agent.

**Self-Attention**
A mechanism used in Transformer models that allows an input to interact with other inputs to determine which parts of the sequence are most important.

**Sequential Model**
A model designed to process sequences of data, like text or time series (e.g., RNNs, LSTMs).

**Sigmoid**
An activation function that squashes its input into a range between 0 and 1, often used for binary classification output layers.

**Softmax**
An activation function that converts a vector of numbers into a probability distribution, where the probabilities of each value are proportional to the relative scale of each value in the vector. Often used for multi-class classification output layers.

**Stride**
The number of pixels a convolutional kernel moves across an image at a time.

**Supervised Learning**
A type of machine learning where the model learns from labeled data.

---
### T

**Tensor**
A multi-dimensional array, which is the primary data structure used in deep learning libraries like TensorFlow and PyTorch.

**TensorFlow**
A popular open-source software library for machine learning and artificial intelligence, developed by Google.

**Tokenization**
The process of breaking down a piece of text into smaller units called tokens (e.g., words or sub-words).

**Transformer**
A revolutionary deep learning architecture based on the self-attention mechanism, which has become the standard for NLP tasks and is increasingly used in computer vision.

**Transfer Learning**
A machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

**Training Set**
The subset of data used to train a model.

**Test Set**
The subset of data used to evaluate the final performance of a trained model.

**Two-Stage Detector**
An object detection model that first proposes regions of interest and then classifies those regions (e.g., Faster R-CNN).

---
### U

**Underfitting**
A problem where a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both the training and test sets.

**Unsupervised Learning**
A type of machine learning where the model learns from unlabeled data.

---
### V

**Validation Set**
The subset of data used during training to tune hyperparameters and evaluate model performance to prevent overfitting.

**Vanishing Gradients**
A problem during training where the gradients become extremely small, preventing the weights in the early layers of a deep network from being updated effectively.

**Variational Autoencoder (VAE)**
A generative model that learns a structured, probabilistic latent space, allowing it to generate new data by sampling from this space.

**Vision Transformer (ViT)**
A model that applies the Transformer architecture directly to sequences of image patches for classification tasks.

**VGGNet**
A very deep CNN architecture that demonstrated that depth was a critical component for performance, using a simple structure of stacked 3x3 convolution layers.

---
### W

**Weights**
The learnable parameters of a neural network that are adjusted during training to minimize the loss function.

**Word2Vec**
A popular technique for learning word embeddings from a text corpus.

---
### X

**Xception**
A CNN architecture that relies entirely on depthwise separable convolution layers, an efficient alternative to standard convolutions.

---
### Y

**YOLO (You Only Look Once)**
A popular one-stage object detection model known for its incredible speed and real-time performance.
