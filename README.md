# Credit-Card-fraud-ML-model

# Robust Ensemble RNN Framework for Adversarial Credit Card Fraud Detection

## Introduction

In today's digital economy, credit card transactions have become increasingly seamless. However, with this convenience comes a heightened risk of fraud. The detection of fraudulent activity is challenging due to the overwhelming number of legitimate transactions, which can obscure the rare instances of fraud. Traditional machine learning models often fail to maintain performance when exposed to adversarial scenarios where data may be intentionally manipulated.

This project proposes a **robust ensemble-based RNN framework** designed for credit card fraud detection under adversarial conditions, with a special focus on **data poisoning attacks**. The model combines various RNN architectures — including LSTM, GRU, and their bidirectional variants — in an ensemble to increase robustness against manipulated training datasets.

Our methodology includes:
- Formatting transaction data into sequences suitable for RNN input,
- Addressing class imbalance via resampling techniques,
- Evaluating performance across both clean and poisoned datasets using metrics such as accuracy, precision, and F1-score.

The project introduces varying degrees of adversarial data perturbations to evaluate how well the ensemble performs under such attacks, contributing insights toward developing more secure and trustworthy fraud detection models.

---

## Motivation

Financial institutions heavily rely on machine learning for detecting fraudulent transactions. Unfortunately, these systems are vulnerable to **adversarial attacks** — especially **data poisoning**, where attackers inject malicious data into the training process to mislead the model.

Key issues addressed include:
- **Data imbalance**: Fraudulent transactions account for less than 0.2% of all transactions.
- **Sophisticated fraud**: Modern attacks are subtle and often escape conventional anomaly detection methods.
- **Robustness**: Models must maintain performance even when the data pipeline is compromised.

This work proposes an **RNN ensemble** that maintains accuracy in clean environments while improving robustness under adversarial attack, with the ultimate goal of safeguarding digital financial ecosystems.

---

## Methodology

### Data Acquisition and Preprocessing

- **Dataset**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions, 492 frauds (0.172%)
- **Features**: Anonymized PCA-transformed features (`V1` to `V28`) + `Amount` and `Time`.

#### Steps:
1. **Exploratory Data Analysis**: Understand feature distributions and identify imbalances.
2. **Stratified Split**: 80% training, 20% testing, maintaining class ratio.
3. **Resampling**: Apply `RandomUnderSampler` to balance the training set while keeping the test set untouched.

---

### Sequence Formation for RNN Input

Since RNNs require sequential data:
- Use **sliding windows** to group transactions into sequences in chronological order.
- Each sequence is shaped as `(batch_size, sequence_length, n_features)`.
- Labels are assigned based on the last transaction in the sequence or the presence of fraud in the window.

---

### Model Architectures

The ensemble consists of the following models:

| Model     | Direction      | Description                                    |
|-----------|----------------|------------------------------------------------|
| LSTM      | Unidirectional | Captures long-term dependencies                |
| GRU       | Unidirectional | Efficient alternative to LSTM                  |
| BiLSTM    | Bidirectional  | Considers past and future context              |
| BiGRU     | Bidirectional  | Lighter bidirectional variant                  |

Each model includes:
- 1–2 recurrent layers
- Hidden sizes: 64 (LSTM/GRU), 128 (BiLSTM/BiGRU)
- Dropout: 20%
- Fully connected layers with Sigmoid activation

**Ensemble Strategy**: Each model outputs a fraud probability, and predictions are averaged across models for final output.

---

### Data Poisoning Strategies

Adversarial attacks used in the study:

1. **Label Flips**:
   - Flip labels of a percentage of non-fraud to fraud and vice versa.
   - Alters the decision boundary.

2. **Feature Perturbation**:
   - Modify feature values of samples to resemble the opposite class.
   - Makes it harder to learn correct patterns.

3. **Backdoor Triggers**:
   - Insert unusual feature patterns into samples.
   - Causes model to misclassify any test sample with that pattern.

> Poisoning is applied *after resampling* and *before training*. 

---

### Training & Evaluation

Two main experiment tracks:

1. **Clean Training**:  
   - Train the ensemble models on resampled, non-poisoned data.
   - Evaluate on clean test data.

2. **Poisoned Training**:  
   - Apply poisoning strategies to the training set.
   - Train using the same hyperparameters and compare performance.


## Future Work

- Explore **defense mechanisms** like adversarial training or data sanitization.
- Evaluate performance using **other public fraud detection datasets** if available.
- Integrate **temporal patterns more deeply** using attention-based mechanisms or Transformer variants.

---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*  
2. Gu, T., Dolan-Gavitt, B., & Garg, S. (2017). BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.  
3. Biggio, B., Nelson, B., & Laskov, P. (2012). Poisoning Attacks against Support Vector Machines.


---

## Contact

**Author**: Samer Meleka  
**Email**: [samermmeleka@gmail.com](mailto:samermmeleka@gmail.com)

---
