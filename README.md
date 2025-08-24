This project demonstrates how to build and train an Artificial Neural Network (ANN) to predict customer churn using the Churn Modelling dataset.

Dataset:

The dataset contains information about bank customers and whether they have churned (exited the bank). The features include:

RowNumber, CustomerId, Surname: Identifiers (not used for training).
CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary: Numerical features.
Geography, Gender: Categorical features.
Exited: Target variable (1 if the customer churned, 0 otherwise).
Libraries Used:

pandas: For data loading and manipulation.
numpy: For numerical operations.
matplotlib.pyplot: For plotting the training history.
tensorflow: For building and training the ANN model.
sklearn: For data splitting, feature scaling, and evaluation metrics.
Methodology:

Data Loading and Exploration: The dataset is loaded using pandas, and the first few rows are displayed to understand the data structure.
Feature Engineering: Categorical features (Geography and Gender) are one-hot encoded using pd.get_dummies. The original categorical columns are dropped from the feature set.
Data Splitting: The dataset is split into training and testing sets using train_test_split from sklearn.model_selection.
Feature Scaling: Numerical features are scaled using StandardScaler from sklearn.preprocessing to ensure that all features have a similar range, which helps the ANN converge faster.
ANN Model Building:
A Sequential model is initialized from tensorflow.keras.models.
Dense layers are added to the model. The first layer is the input layer, followed by hidden layers with relu activation. Dropout layers are included after the input layer to prevent overfitting.
The output layer has a single unit with sigmoid activation, suitable for binary classification.
Model Compilation: The model is compiled with the Adam optimizer, binary_crossentropy loss function (for binary classification), and accuracy as the evaluation metric.
Early Stopping: EarlyStopping callback from tensorflow.keras.callbacks is used to monitor the validation loss and stop training if it stops improving, preventing overfitting.
Model Training: The model is trained on the training data using the fit method. A validation split is used to monitor performance on unseen data during training.
Model Evaluation:
Predictions are made on the test set using the trained model.
A confusion matrix is generated using confusion_matrix from sklearn.metrics to evaluate the model's performance.
The accuracy score is calculated using accuracy_score from sklearn.metrics.
Training history (accuracy and loss) is plotted to visualize the model's learning process.
This project provides a comprehensive example of applying ANNs to a real-world classification problem, including data preprocessing, model building, training, and evaluation.
