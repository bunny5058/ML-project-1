# California Housing Price Prediction Project

A comprehensive machine learning project that predicts California housing prices using multiple regression techniques and follows best practices for data preprocessing, model evaluation, and hyperparameter tuning.

## Project Overview

This project implements a complete machine learning pipeline for predicting median house values in California using the California Housing dataset. The project demonstrates:

- **Data preprocessing** with proper handling of missing values and feature scaling
- **Stratified sampling** to ensure representative train/test splits
- **Multiple regression algorithms** including Linear Regression, Decision Trees, and Random Forest
- **Cross-validation** for robust model evaluation
- **Hyperparameter tuning** using GridSearchCV
- **Final model evaluation** on holdout test data

## Dataset

The project uses the **California Housing dataset** which contains information about:
- Median house values for California districts
- 8 features including median income, house age, average rooms, population, etc.
- Data derived from the 1990 California census

**Dataset Statistics:**
- **Instances:** 20,640
- **Features:** 8 numerical features + 1 categorical feature (ocean_proximity)
- **Target:** median_house_value

## Project Structure

```
├── keras_ml.py              # Main Python script with complete ML pipeline
├── keras_ML.ipynb           # Jupyter notebook version
├── magic04.data             # MAGIC gamma telescope dataset (unused in current project)
├── magic04.names            # Dataset description for MAGIC telescope data
├── housing_histograms.png   # Generated visualization (when script is run)
├── datasets/                # Directory for housing dataset
└── README.md               # This file
```

## Installation & Requirements

### Prerequisites
- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

### Installation
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

### Running the Complete Pipeline

```bash
python keras_ml.py
```

The script will automatically:
1. Load and explore the California housing dataset
2. Generate and save histograms for data visualization
3. Perform stratified train/test split
4. Apply preprocessing pipeline (imputation + scaling + encoding)
5. Train and evaluate multiple models using cross-validation
6. Fine-tune the best performing model
7. Provide final evaluation on test set

### Expected Output
```
Loading data...
Dataset shape: (20640, 9)
Data head:
   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
0    -122.23     37.88                41.0        880.0           129.0

...

Final RMSE on Test Set: $47,730.45
```

## Machine Learning Pipeline

### 1. Data Loading & Exploration
- Loads housing data from CSV
- Displays dataset information and statistics
- Generates histograms for all numerical features

### 2. Data Preprocessing
- **Stratified Split**: Uses income categories to ensure balanced train/test sets
- **Feature Engineering**: Creates income categories for stratified sampling
- **Preprocessing Pipeline**:
  - Numerical features: Median imputation + Standard scaling
  - Categorical features: One-hot encoding for ocean_proximity

### 3. Model Training & Evaluation
- **Cross-validation** with 10 folds for robust evaluation
- **Multiple algorithms**:
  - Linear Regression (baseline)
  - Decision Tree Regressor
  - Random Forest Regressor (best performer)

### 4. Hyperparameter Tuning
- Uses GridSearchCV to optimize Random Forest parameters
- Tests multiple combinations of n_estimators and max_features
- 5-fold cross-validation for parameter selection

### 5. Final Evaluation
- Evaluates best model on holdout test set
- Reports final RMSE score

## Model Performance

### Cross-Validation Results (RMSE):
- **Linear Regression**: ~$68,600 ± $2,500
- **Decision Tree**: ~$71,400 ± $2,800
- **Random Forest**: ~$50,200 ± $1,800

### Final Test Performance:
- **Best Model (Tuned Random Forest)**: ~$47,730 RMSE

## Key Techniques Used

### 1. Stratified Sampling
- Addresses sampling bias by ensuring income distribution matches between train/test sets
- Uses `StratifiedShuffleSplit` from scikit-learn

### 2. Feature Preprocessing Pipeline
- **ColumnTransformer**: Applies different preprocessing to numerical vs categorical features
- **SimpleImputer**: Handles missing values using median strategy
- **StandardScaler**: Normalizes numerical features
- **OneHotEncoder**: Converts categorical ocean_proximity to numerical

### 3. Model Selection & Evaluation
- **Cross-validation**: 10-fold CV for reliable performance estimates
- **RMSE**: Primary metric (Root Mean Squared Error)
- **GridSearchCV**: Systematic hyperparameter optimization

### 4. Ensemble Methods
- **Random Forest**: Best performing algorithm using ensemble of decision trees
- **Bootstrap aggregating (bagging)**: Reduces overfitting

## File Structure Details

### `keras_ml.py`
Main script containing:
- `display_scores()`: Helper function for cross-validation results
- `main()`: Complete ML pipeline from data loading to final evaluation

### `keras_ML.ipynb`
Jupyter notebook version with same functionality, useful for:
- Interactive exploration
- Step-by-step execution
- Visualization within notebook

## Future Improvements

1. **Feature Engineering**:
   - Add geographical features (distance to cities, coast)
   - Create ratio features (bedrooms/rooms, population/households)

2. **Advanced Models**:
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks (TensorFlow/Keras)
   - Support Vector Regression

3. **Model Interpretability**:
   - Feature importance analysis
   - Partial dependence plots
   - SHAP values for explainability

4. **Production Deployment**:
   - Model serialization with pickle/joblib
   - REST API with Flask/FastAPI
   - Docker containerization

## References & Credits

- **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** by Aurélien Géron
- **California Housing Dataset** from StatLib repository
- **Scikit-learn documentation** for implementation details

## License

This project is for educational purposes. The California Housing dataset is in the public domain.

## Contributing

Feel free to submit issues and enhancement requests!
