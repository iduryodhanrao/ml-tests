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


---

# 📊 Online vs. Offline Metrics in Machine Learning

Understanding the distinction between **offline** and **online** evaluation metrics is key to building robust, production-ready ML systems.
- **Offline metrics** help validate model performance before deployment.
- **Online metrics** measure real-world impact and user behavior post-deployment.
- A robust ML workflow uses **both** to ensure reliability, relevance, and business value.

---

## 🧪 Offline Metrics

Offline metrics are computed using **historical or hold-out datasets** in a controlled environment. They help assess model performance before deployment.

| Metric                     | Description                                                                 | Real-World Model Example              | Applicability |
|---------------------------|-----------------------------------------------------------------------------|--------------------------------------|---------------|
| Accuracy                  | Proportion of correct predictions                                           | Email spam classifier (LogReg, CNN)  | Both          |
| Precision                 | True positives / predicted positives                                        | Fraud detection (XGBoost, BERT)      | Both          |
| Recall (Sensitivity)      | True positives / actual positives                                           | Cancer detection (DT, ResNet)        | Both          |
| F1 Score                  | Harmonic mean of precision and recall                                       | Churn prediction (SVM, LSTM)         | Both          |
| ROC-AUC                   | Area under ROC curve; measures ranking quality                              | Credit risk scoring (RF, DNN)        | Both          |
| Log Loss                  | Penalizes incorrect probability estimates                                   | Sentiment analysis (Softmax Classifier) | Both        |
| RMSE / MAE                | Regression errors (Root Mean Squared / Mean Absolute Error)                 | House price prediction (Linear, LSTM)| Both          |
| R² (Coefficient of Determination) | Proportion of variance explained by the model                     | Marketing ROI modeling (Regression)  | NDL           |
| NDCG / MAP                | Ranking quality in recommendation systems                                   | Product ranking (GBoost Ranker, Transformer) | Both    |
| Confusion Matrix          | Breakdown of TP, FP, FN, TN                                                 | Disease classification (CNN)         | Both          |

---

## 🌐 Online Metrics

Online metrics are collected **after deployment**, often via **A/B testing** or **live user interactions**. They reflect real-world impact.

| Metric                     | Description                                                                 | Real-World Model Example                | Applicability |
|---------------------------|-----------------------------------------------------------------------------|----------------------------------------|---------------|
| Click-Through Rate (CTR)  | Ratio of clicks to impressions                                              | News recommendation (LogReg, Wide&Deep) | Both         |
| Conversion Rate           | Percentage of users completing a desired action                             | Ad targeting (GBM, Transformer)         | Both          |
| Bounce Rate               | Percentage of users leaving without interaction                             | Search ranking algorithm (SVM, RankNet)| NDL           |
| Dwell Time                | Time spent engaging with content                                             | Video recommendation (RNN, BERT4Rec)   | DL            |
| Retention Rate            | Users returning after initial interaction                                   | Mobile personalization (Deep Seq Model)| DL            |
| Revenue Lift              | Increase in revenue due to model deployment                                 | Dynamic pricing (Elastic Net, DNN)     | Both          |
| Engagement Score          | Composite metric of user interactions                                       | Feed ranking (CNN + RNN Features)      | DL            |
| Satisfaction Score (DSAT) | Explicit user feedback or dissatisfaction analytics                         | Voice assistant (Sentiment Classifier) | DL            |
| Latency / Throughput      | Model response time and scalability                                          | Fraud detection (Real-time LSTM)       | DL            |
| Model Drift Detection     | Monitoring distribution shift over time                                     | Predictive maintenance (Monitoring + DL) | DL         |

---

## 🔍 Notes

- **DL**: Models like CNNs, RNNs, Transformers.
- **NDL**: Classical ML models — Logistic Regression, Trees, SVMs.
- **Both**: Metrics useful across model types depending on task complexity and data modality.
