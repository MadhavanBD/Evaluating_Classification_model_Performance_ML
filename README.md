This repository contains code for analyzing a digital advertising dataset using logistic regression and evaluating the model's performance using various metrics,
including the Confusion Matrix, Accuracy Score, Receiver Operating Characteristic (ROC) Curve, Cross Validation Score, and Cumulative Accuracy Profile (CAP) Curve.

Dataset
The digital advertising dataset used in this analysis is loaded from a CSV file using the pandas library. The dataset contains information about various features related 
to digital advertising, such as ad impressions, clicks, and website visits.

Logistic Regression Model
A logistic regression model is trained on the dataset to predict the effectiveness of digital ads. The dataset is split into training and testing sets using the train_test_split 
function from scikit-learn. The features are standardized using the StandardScaler from scikit-learn to ensure consistent scaling across different features.

Model Evaluation : 

Confusion Matrix
The Confusion Matrix is computed using the confusion_matrix function from scikit-learn. It provides a tabular representation of the model's predictions compared to the actual labels. 
The matrix shows the number of true positives, true negatives, false positives, and false negatives.

Accuracy Score
The Accuracy Score is calculated using the accuracy_score function from scikit-learn. It measures the percentage of correctly predicted labels out of the total number of predictions. 
The accuracy score provides an overall measure of the model's performance.

Receiver Operating Characteristic (ROC) Curve
The ROC Curve is generated using the roc_curve and roc_auc_score functions from scikit-learn. It visualizes the trade-off between the true positive rate and the false positive rate at 

various classification thresholds. The area under the ROC curve (AUC) is also computed to quantify the model's performance.

Cross Validation Score
The Cross Validation Score is calculated using the cross_val_score function from scikit-learn. It performs k-fold cross-validation on the dataset to evaluate the model's performance. 
Both KFold and StratifiedKFold methods are used to ensure reliable evaluation results.

Cumulative Accuracy Profile (CAP) Curve
The CAP Curve is plotted to assess the model's performance in classifying positive observations. The cumulative number of positive observations is plotted against the total number of
observations. The CAP Curve is compared to the Random Model (a baseline) and the Perfect Model (ideal performance). The CAP value at the 50% mark indicates the model's accuracy in 
identifying positive observations.

Usage
To replicate the analysis, follow these steps:
Ensure that you have the necessary dependencies installed, including pandas, numpy, scikit-learn, and matplotlib.
Load the digital advertising dataset from the CSV file using the appropriate file path.
Train a logistic regression model on the dataset and split it into training and testing sets.
Standardize the feature variables using the StandardScaler.
Evaluate the model's performance using the Confusion Matrix, Accuracy Score, ROC Curve, Cross Validation Score, and CAP Curve.
Modify the code as needed to adapt to your specific dataset and requirements.

Conclusion
The analysis of the digital advertising dataset using logistic regression provides valuable insights into the effectiveness of digital ads. By evaluating various metrics, 
such as the Confusion Matrix, Accuracy Score, ROC Curve, Cross Validation Score, and CAP Curve, we can assess the model's performance and make informed decisions 
about digital advertising strategies.




