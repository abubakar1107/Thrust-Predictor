# README: Thrust Prediction Using Neural Networks

## Overview

This script performs exploratory data analysis (EDA) and builds neural network models using TensorFlow/Keras to predict thrust based on RPM, Diameter, and Pitch. The dataset is loaded from `csv.csv` and consists of `Thrust(lb)`, `RPM`, `Diameter`, and `PITCH`.

### Key Steps

1. **Exploratory Data Analysis (EDA)**:
   - Load the dataset using `pandas`.
   - Visualize thrust distribution and identify errors, dropping rows with invalid thrust values.
   - Compute summary statistics and a correlation matrix to assess relationships between features.
   - Visualize data using histograms, scatter plots, and Seaborn's `pairplot`.

2. **Data Preprocessing**:
   - Handle missing values by dropping rows with `NaN`.
   - Split the dataset into training and testing sets using `train_test_split`.
   - Scale the features using `MinMaxScaler`.

3. **Model Building**:
   - Two neural network models (`model_1` and `model_2`) are built using the Sequential API from Keras. 
   - The models use `relu` activation functions and `Dropout` for regularization.
   - Both models are compiled with the Adam optimizer and mean absolute error (MAE) as the loss function.
   - Early stopping is implemented to prevent overfitting by monitoring validation loss.

4. **Model Training**:
   - Models are trained on the dataset with a callback to stop training when the validation loss plateaus.
   - Model 1 has fewer layers, while Model 2 is more complex, aiming to improve performance.

5. **Model Evaluation**:
   - Evaluate performance using Mean Absolute Error (MAE) and Explained Variance Score.
   - Visualize true vs. predicted values using scatter plots.
   - The first model shows good performance for lower thrust values but deviates at higher values.

6. **Saving the Model**:
   - Save the trained model using `model.save('thrust_predicter_1.h5')`.

### Key Libraries
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow.keras`, `sklearn`

## Conclusion

This script demonstrates data exploration, cleaning, and modeling for thrust prediction. The final models perform reasonably well and can be further tuned for improved accuracy.
