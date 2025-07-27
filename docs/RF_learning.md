# 🧠 Deep Learning Concepts: NLP & Reinforcement Learning

This document provides a detailed overview of foundational concepts in Natural Language Processing (NLP) and Reinforcement Learning (RL), from word embeddings to advanced learning algorithms.

---
## 🔡 Part 1: Neural Embeddings & NLP Models

Neural embeddings are dense vector representations of words, sentences, or other data types. They capture semantic relationships, allowing models to understand context.

### Foundational Embedding Models

| Model | Core Idea | Use Case / Example | Pros & Cons |
| :--- | :--- | :--- | :--- |
| **Word2Vec** | Learns word associations from a large text corpus. It has two main architectures: <br> 1. **CBOW (Continuous Bag-of-Words):** Predicts a target word based on its surrounding context words. <br> 2. **Skip-gram:** Predicts the surrounding context words given a target word. | **Semantic Search:** Finding words with similar meanings (e.g., "king" - "man" + "woman" ≈ "queen"). Used in search engines and recommendation systems. | **Pros:** Computationally efficient, captures semantic relationships well. <br> **Cons:** Cannot handle out-of-vocabulary words, does not account for polysemy (words with multiple meanings). |
| **GloVe** | **Global Vectors for Word Representation.** Learns embeddings by analyzing a global word-word co-occurrence matrix from a corpus. It aims to combine the strengths of matrix factorization and local context window methods. | **Text Classification:** Initializing the first layer of a neural network for sentiment analysis or topic classification with pre-trained GloVe vectors. | **Pros:** Often performs better than Word2Vec on word analogy tasks, trains on global statistics. <br> **Cons:** Same limitations as Word2Vec regarding unknown words and polysemy. |
| **Transformers** | A powerful architecture that generates **contextual embeddings**. Unlike Word2Vec or GloVe, the embedding for a word changes based on the sentence it's in. It uses a **self-attention mechanism** to weigh the importance of all other words in the input when processing a specific word. | **Machine Translation & Summarization:** Models like BERT and GPT use Transformers to understand nuanced language, leading to state-of-the-art performance in translation and text generation. | **Pros:** Captures context and polysemy perfectly, state-of-the-art performance on most NLP tasks. <br> **Cons:** Very computationally expensive to train and run. |

### Sequential Models

These models are designed to process sequences of data, like text or time series.

| Model | Core Idea | Use Case / Example | Pros & Cons |
| :--- | :--- | :--- | :--- |
| **Recurrent Neural Networks (RNNs)** | A type of neural network with a "memory" loop. The output from a previous step is fed as input to the current step, allowing it to maintain a **hidden state** or context as it processes a sequence. | **Basic Language Modeling:** Predicting the next character or word in a sequence. | **Pros:** Simple architecture, can process sequences of any length. <br> **Cons:** Suffers from the **vanishing/exploding gradient problem**, making it difficult to learn long-range dependencies. |
| **Long Short-Term Memory (LSTMs)** | An advanced type of RNN designed to overcome the vanishing gradient problem. It uses a series of "gates" (input, forget, and output gates) to regulate the flow of information, allowing it to remember or forget information over long sequences. | **Sentiment Analysis of Reviews:** Analyzing a long movie or product review to determine the overall sentiment, even if key context appeared early in the text. | **Pros:** Excellent at capturing long-term dependencies in sequences. <br> **Cons:** More complex and computationally intensive than simple RNNs. |

### NLP Tasks

| Task | Core Idea | Use Case / Example |
| :--- | :--- | :--- |
| **Named Entity Recognition (NER)** | An information extraction task that seeks to locate and classify named entities (pre-defined categories like names of persons, organizations, locations, dates, etc.) in unstructured text. | **Content Analysis:** A news organization uses NER to automatically scan articles and tag all mentions of companies, people, and places for easier indexing and search. |

---
## 🤖 Part 2: Reinforcement Learning (RL)

Reinforcement Learning is an area of machine learning where an **agent** learns to make decisions by performing actions in an **environment** to maximize a cumulative **reward**.

### Core RL Algorithms

| Algorithm | Core Idea | Use Case / Example | Pros & Cons |
| :--- | :--- | :--- | :--- |
| **Q-learning** | An **off-policy**, value-based algorithm. It learns the value of taking a certain action in a certain state (the "Q-value"). It directly approximates the optimal Q-value, regardless of the policy being followed during training, by always choosing the action with the maximum Q-value in the next state. | **Simple Games:** Training an agent to play a simple grid-world game where it must find the optimal path to a goal while avoiding obstacles. | **Pros:** Simple to implement, guaranteed to converge to the optimal policy if all state-action pairs are explored. <br> **Cons:** Does not work well with very large or continuous state spaces. |
| **SARSA** | **State-Action-Reward-State-Action.** An **on-policy**, value-based algorithm. It is very similar to Q-learning, but instead of choosing the action with the maximum Q-value for the next step, it updates its Q-value based on the action that was *actually* taken by the current policy. | **Robotics:** Training a robot to navigate a path where safety is critical. SARSA tends to be more conservative and avoids risky actions because it learns from what it actually does. | **Pros:** Learns a "safer" policy by accounting for exploration. <br> **Cons:** Can be less efficient at finding the optimal policy compared to Q-learning. |

### Foundational RL Methods

| Method | Core Idea |
| :--- | :--- |
| **Dynamic Programming** | A collection of algorithms that can compute optimal policies given a perfect model of the environment (i.e., knowing the exact probabilities of transitions and rewards). It uses value functions to structure the search for good policies. |
| **Monte Carlo Methods** | Model-free methods that learn directly from episodes of experience. An agent runs through an entire episode, and only then are the value functions for the visited states updated based on the total reward received. |

### Key Concepts in RL

| Concept | Tabular Approach | Function Approximation Approach |
| :--- | :--- | :--- |
| **Description** | Represents the value function (e.g., Q-values) as a **table or matrix**. There is one entry for every possible state or state-action pair. | Represents the value function using a **parameterized function**, such as a neural network. The network takes a state as input and outputs the estimated value. |
| **Use Case** | Environments with a small, discrete number of states and actions, like Tic-Tac-Toe or simple grid worlds. | Environments with very large or continuous state spaces, like the game of Go, controlling a robot arm, or playing video games from pixel inputs. |
| **Pros** | Simple, exact, and guaranteed to find the optimal solution. | Can handle huge state spaces, can generalize to unseen states. |
| **Cons** | Becomes completely infeasible as the number of states grows (the "curse of dimensionality"). Cannot handle continuous states. | The learning process can be unstable, and convergence to the optimal solution is not guaranteed. |
