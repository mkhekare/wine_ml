# Wine Quality Prediction

## Overview

This project aims to predict the quality of wine using various features from two datasets: red and white wine. It addresses the class imbalance problem prevalent in the dataset, particularly in the quality labels.

## Problem Statement

The main goal is to predict wine quality based on various chemical properties. The dataset consists of features such as acidity, sugar content, and alcohol level, among others.

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning models and metrics.
- **Imbalanced-learn**: For handling class imbalance using SMOTE.
- **Seaborn & Matplotlib**: For data visualization.

## Steps in the Project

1. **Data Loading**:
   - Load the datasets for red and white wine.
  
   ```python
   import pandas as pd
   white_wine = pd.read_csv('winequality-white.csv', sep=';')
   red_wine = pd.read_csv('winequality-red.csv', sep=';')
   ```

2. **Data Preparation**:
   - Add a feature indicating the type of wine (red or white).
   - Merge the two datasets and shuffle the observations.
   - Create a quality label based on the quality score.

3. **Data Exploration**:
   - Visualize the distribution of wine quality labels to identify class imbalance.
  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.countplot(x=wines['quality_label'])
   plt.show()
   ```

4. **Data Splitting**:
   - Split the data into training and test sets.

5. **Data Scaling**:
   - Scale the features using `StandardScaler`.

6. **Model Training**:
   - Train a Logistic Regression model on the imbalanced dataset.
  
   ```python
   from sklearn.linear_model import LogisticRegression
   lg = LogisticRegression()
   lg.fit(X_train, y_train)
   ```

7. **Handling Class Imbalance**:
   - Apply SMOTE to balance the classes in the training set.
  
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE()
   X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
   ```

8. **Model Evaluation**:
   - Predict on the test set and evaluate the model using confusion matrix and classification report.
  
   ```python
   y_pred_smote = lg.predict(X_test)
   from sklearn.metrics import confusion_matrix, classification_report
   results = confusion_matrix(y_test, y_pred_smote)
   print("Confusion Matrix:\n", results)
   print("Classification Report:\n", classification_report(y_test, y_pred_smote))
   ```

## Results

- The confusion matrix and classification report will provide insights into the model's performance, particularly in predicting the minority class.

## Conclusion

This project demonstrates the importance of addressing class imbalance in predictive modeling. By applying SMOTE, we can improve the model's ability to predict underrepresented classes effectively.

## Future Work

- Experiment with other machine learning algorithms to further improve prediction accuracy.
- Implement hyperparameter tuning for better model performance.
- Explore additional feature engineering techniques to enhance the dataset.

## Acknowledgments

- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from the UCI Machine Learning Repository.
- Various libraries and frameworks that facilitate data science and machine learning tasks.

## License

This project is licensed under the MIT License.
