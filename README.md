# Credit Card Fraud Detection Using Machine Learning and Deep Learning

## Project Overview

This project aims to develop an effective credit card fraud detection system using various machine learning and deep learning techniques. We explore different models and sampling methods to address the challenge of highly imbalanced data in fraud detection.

### What Was Done

- **Data Exploration**: Analyzed the credit card transaction dataset to understand its characteristics and the extent of class imbalance.
- **Data Preprocessing**: Handled data normalization and feature scaling.
- **Model Development**: Implemented and compared multiple approaches:
  - Random Forest
  - Decision Trees
  - Neural Networks (with different sampling techniques)
- **Sampling Techniques**: Applied various methods to address class imbalance:
  - Weighted loss
  - Undersampling
  - Oversampling using SMOTE (Synthetic Minority Over-sampling Technique)
- **Performance Evaluation**: Assessed models using metrics such as accuracy, precision, recall, and F1-score.

### Key Findings

- The dataset is highly imbalanced, with fraudulent transactions making up less than 0.1% of all transactions.
- Simple accuracy is not a suitable metric due to the class imbalance.
- Oversampling using SMOTE combined with a neural network provided the best results, achieving 100% recall on fraudulent transactions.

## Skills Demonstrated

- Data Analysis and Preprocessing
- Machine Learning (Random Forest, Decision Trees)
- Deep Learning (Neural Networks)
- Handling Imbalanced Datasets
- Model Evaluation and Comparison


## Dataset

You can download the dataset from the following Dropbox link:
[Credit Card Fraud Detection Dataset](https://www.dropbox.com/scl/fi/7umj4vjaprft83guxsczj/creditcard.csv?rlkey=3n1wlw6y13iv8t1zyqee2ow3o&st=tdsxqgya&dl=0)

The dataset used in this project contains credit card transactions made over a two-day period, with 492 frauds out of 284,807 transactions. Key features include:

- `Time`: Seconds elapsed between each transaction and the first transaction
- `Amount`: Transaction amount
- `V1-V28`: Anonymized features (result of a PCA transformation)
- `Class`: Target variable (1 for fraud, 0 for normal transaction)

### Data Characteristics
- Highly unbalanced: Only 0.172% of transactions are fraudulent
- All numerical input variables
- No missing values
- 
## Methodology

### Data Preprocessing
1. Normalized the 'Amount' feature using StandardScaler
2. Removed the 'Time' feature as it wasn't relevant for the models

### Model Development
We implemented and compared several models:

1. **Random Forest**
   - Used 100 estimators
   - Achieved high accuracy but struggled with recall on fraudulent transactions

2. **Decision Trees**
   - Simple implementation
   - Showed decent performance but was outperformed by Random Forest

3. **Neural Networks**
   - Implemented using Keras with TensorFlow backend
   - Tested various configurations:
     a. Basic neural network
     b. Neural network with weighted loss
     c. Neural network with undersampled data
     d. Neural network with SMOTE oversampling

### Addressing Class Imbalance
We explored multiple techniques to handle the severe class imbalance:

1. **Weighted Loss Function**: Assigned higher weight to the minority class
2. **Undersampling**: Reduced majority class samples to match minority class
3. **SMOTE Oversampling**: Created synthetic samples of the minority class

Each technique was evaluated based on its impact on model performance, particularly recall for fraudulent transactions.

## Challenges and Solutions

### 1. Extreme Class Imbalance
**Challenge**: With only 0.172% fraudulent transactions, standard models were biased towards the majority class.
**Solution**: Implemented SMOTE oversampling, which proved most effective in balancing class representation.

### 2. Model Overfitting
**Challenge**: Initial neural network models showed signs of overfitting to the training data.
**Solution**: Introduced dropout layers and early stopping to improve generalization.

### 3. Performance Metric Selection
**Challenge**: Accuracy alone was misleading due to class imbalance.
**Solution**: Focused on recall, precision, and F1-score, with emphasis on minimizing false negatives.

### 4. Computational Resources
**Challenge**: SMOTE increased dataset size significantly, straining computational resources.
**Solution**: Utilized batch processing and optimized model architectures for efficiency.

## Results and Discussion

### Model Performance Comparison

| Model | Accuracy | False Neg Rate | Recall | Precision | F1 Score |
|-------|----------|----------------|--------|-----------|----------|
| Random Forest | 0.999860 | 0.069106 | 0.930894 | 0.987069 | 0.958159 |
| Decision Tree | 0.999772 | 0.073171 | 0.926829 | 0.940206 | 0.933470 |
| Basic Neural Network | 0.999403 | 0.211382 | 0.788618 | 0.854626 | 0.820296 |
| Weighted NN | 0.976349 | 0.048780 | 0.951220 | 0.065181 | 0.122002 |
| Undersampled NN | 0.965801 | 0.032520 | 0.967480 | 0.046667 | 0.089039 |
| SMOTE NN | 0.997353 | 0.000000 | 1.000000 | 0.394864 | 0.566168 |

### Key Observations

1. **SMOTE Neural Network**: Achieved perfect recall (1.0) for fraudulent transactions, crucial for fraud detection.
2. **Trade-off**: While SMOTE NN had lower precision, the importance of catching all frauds outweighs the cost of investigating false positives.
3. **Traditional Models**: Random Forest and Decision Trees showed good overall performance but missed more fraudulent cases compared to the SMOTE NN.
4. **Basic NN vs Enhanced Techniques**: Significant improvement observed when addressing class imbalance, particularly with SMOTE.

### Implications
The results demonstrate the power of combining deep learning with advanced sampling techniques in handling imbalanced datasets. The SMOTE Neural Network's ability to detect all fraudulent transactions makes it a valuable tool for real-world fraud detection systems, where missing a fraud can be extremely costly.

## Visualizations

### 1. Correlation with Class (Fraudulent or Not)

<div align="center">
    <img width="644" alt="Screenshot 2024-08-06 at 4 28 28 AM" src="https://github.com/user-attachments/assets/5bcb50e0-c3db-463e-93f8-748b263a5a05">
    <p><strong>Feature Correlation with Fraudulent Transactions</strong></p>
</div>

#### Description:
This bar chart shows the correlation of various features (V1-V28) with the target variable (fraudulent or not).

**Findings**: 
- Several features (e.g., V17, V14, V12) show strong negative correlation with fraud.
- Some features (e.g., V4, V11) show positive correlation with fraud.
- This information is crucial for feature selection and understanding fraud indicators.

### 2. ROC Curve for Random Forest Classifier

<div align="center">
    <img width="337" alt="Screenshot 2024-08-06 at 4 29 45 AM" src="https://github.com/user-attachments/assets/c009c61c-07a0-405d-8dad-57ca72d35421">
    <p><strong>ROC Curve for Random Forest Classifier</strong></p>
</div>

#### Description:
This plot shows the Receiver Operating Characteristic (ROC) curve for the Random Forest model.

**Findings**:
- The Random Forest model achieves a high Area Under the Curve (AUC) of 0.941.
- This indicates strong discriminative ability between fraudulent and non-fraudulent transactions.

### 3. Model Performance Comparison

<div align="center">
    <img width="468" alt="Screenshot 2024-08-06 at 4 30 39 AM" src="https://github.com/user-attachments/assets/0f7bb4ea-54b5-4bfd-be61-bcc4a94c5d66">
    <p><strong>Comparison of Different Models and Techniques</strong></p>
</div>


#### Description:
This table compares the performance metrics of different models used in the project.

**Findings**:
- The OverSampledNeuralNetwork achieves the best recall (1.000000) and high accuracy (0.997353).
- Traditional methods like Random Forest perform well but are outperformed by neural networks in detecting fraudulent transactions.
- Weighted and undersampled neural networks show improvements over the plain neural network.

### 4. Confusion Matrix for Best Model

<div align="center">
    <img width="285" alt="Screenshot 2024-08-06 at 4 31 19 AM" src="https://github.com/user-attachments/assets/d41cb888-459c-41cc-9215-bf68e05643ed">
    <p><strong>Confusion Matrix of Best Performing Model on Full Dataset</strong></p>
</div>

#### Description:
This confusion matrix visualizes the performance of the best model (likely the OverSampledNeuralNetwork) on the full dataset.

**Findings**:
- The model correctly identifies all 492 fraudulent transactions (100% recall).
- Only 754 non-fraudulent transactions are incorrectly flagged as fraudulent.
- This demonstrates excellent performance in fraud detection while maintaining a low false positive rate.

## How to Use

To explore and utilize this project:

1. **Clone the Repository**:
   ```
   git clone https://github.com/crishN144/Credit-Card-Fraud-Detection-System.git
   cd Credit-Card-Fraud-Detection-System
   ```

2. **Set Up the Environment**:
   - It's recommended to use a virtual environment:
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```
   - Install required dependencies:
     ```
     pip install -r requirements.txt


     ```
   Note: If requirements.txt is not provided, you may need to install necessary libraries manually (numpy, pandas, scikit-learn, keras, tensorflow, imbalanced-learn, matplotlib, seaborn).

3. **Run the Notebooks**:
   - Jupyter notebooks are provided for data exploration, model training, and evaluation.
   - Launch Jupyter notebook:
     ```
     jupyter notebook
     ```
   - Open and run the notebooks in the provided sequence.

4. **Experiment and Modify**:
   - Feel free to modify the models, parameters, and techniques.
   - Experiment with different architectures and sampling methods to see their effects on model performance.

## References
- Machine Learning and Deep Learning libraries: scikit-learn, keras, tensorflow, imbalanced-learn
- Visualization libraries: matplotlib, seaborn
