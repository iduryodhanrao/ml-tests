# 🔧 Common Hyperparameters to Tune by Model Category

## 🧠 Non-Deep Learning Models

| Model Name                | Common Hyperparameters to Tune                                                                 | Real-World Example |
|--------------------------|-----------------------------------------------------------------------------------------------|--------------------|
| Logistic Regression       | `solver`, `penalty` (L1, L2, Elastic-net), `C` (regularization strength), `class_weight`     | Credit scoring and churn prediction |
| K-Nearest Neighbors (KNN) | `n_neighbors`, `weights` (uniform, distance), `metric` (Euclidean, Manhattan, cosine)        | Recommender systems for retail (e.g., product similarity) |
| Support Vector Machines   | `kernel` (linear, poly, rbf, sigmoid), `C`, `gamma`, `epsilon` (for SVR), `class_weight`     | Image classification (e.g., handwritten digit recognition) |
| Decision Trees            | `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`, `splitter` | Loan approval and risk modeling |
| Random Forest             | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `criterion` | Fraud detection in banking |
| Gradient Boosting (XGBoost, LightGBM) | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `lambda`, `alpha`, `scale_pos_weight` | Customer segmentation and click-through rate prediction |
| Naive Bayes               | `var_smoothing` (Gaussian NB), `alpha` (Multinomial NB)                                      | Spam filtering and document classification |
| Linear Discriminant Analysis (LDA) | `solver`, `shrinkage`, `n_components`                                               | Face recognition and dimensionality reduction |
| Ridge/Lasso Regression    | `alpha`, `fit_intercept`, `normalize`                                                       | House price prediction and marketing ROI modeling |

## 🤖 Deep Learning Models

| Model Name                  | Common Hyperparameters to Tune                                                                 | Real-World Example |
|----------------------------|-----------------------------------------------------------------------------------------------|--------------------|
| Convolutional Neural Networks (CNN) | `learning_rate`, `batch_size`, `epochs`, `optimizer`, `activation_function`, `dropout_rate`, `number_of_layers`, `neurons_per_layer`, `kernel_size`, `number_of_filters`, `pooling_size` | Medical image analysis (e.g., tumor detection in MRIs) |
| Recurrent Neural Networks (RNN)     | `learning_rate`, `hidden_units`, `batch_size`, `epochs`, `dropout_rate`, `number_of_layers`, `RNN_cell_architecture` (LSTM, GRU), `weight_initialization_strategy`, `gradient_clipping` | Time series forecasting (e.g., stock prices, energy demand) |
| Generative Adversarial Networks (GAN) | `learning_rate` (generator & discriminator), `batch_size`, `epochs`, `optimizer`, `loss_functions`, `network_architecture`, `regularization`, `activation_function`, `weight_initialization` | Synthetic image generation (e.g., fashion design, art synthesis) |
| Transformer Models (e.g., BERT, GPT) | `learning_rate`, `batch_size`, `epochs`, `warmup_steps`, `max_seq_length`, `num_attention_heads`, `dropout_rate`, `weight_decay` | Sentiment analysis, question answering, document summarization |
| Autoencoders                | `latent_dim`, `activation_function`, `optimizer`, `epochs`, `batch_size`, `dropout_rate`     | Anomaly detection in network traffic or manufacturing |
| Deep Reinforcement Learning | `learning_rate`, `discount_factor (gamma)`, `epsilon`, `batch_size`, `target_update_freq`, `replay_buffer_size` | Autonomous driving and robotic control |
