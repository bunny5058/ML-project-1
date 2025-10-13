import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def display_scores(scores):
    """A helper function to display cross-validation scores."""
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def main():
    """Main function to run the machine learning pipeline."""
    # --- 1. Load the Data ---
    print("Loading data...")
    housing = pd.read_csv("housing.csv")
    print("Dataset shape:", housing.shape)
    print("Data head:\n", housing.head())
    print("\n")

    # --- 2. Initial Data Exploration ---
    print("Data Info:")
    housing.info()
    print("\nData Description:\n", housing.describe())
    print("\n")

    # --- 3. Visualize the Data ---
    print("Generating and saving histograms...")
    housing.hist(bins=50, figsize=(20, 15))
    plt.savefig("housing_histograms.png")
    plt.close()
    print("Histograms saved to housing_histograms.png")
    print("\n")

    # --- 4. Create Income Categories for Stratified Splitting ---
    print("Creating income categories for stratified splitting...")
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    # --- 5. Perform Stratified Split ---
    print("Performing stratified split...")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    print(f"Training set size: {len(strat_train_set)}")
    print(f"Test set size: {len(strat_test_set)}")
    print("\n")

    # --- 6. Separate Features and Labels ---
    print("Separating features and labels...")
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    print("\n")

    # --- 7. Define and Run the Preprocessing Pipeline ---
    print("Defining and running the preprocessing pipeline...")
    num_attribs = list(housing.select_dtypes(include=np.number).columns)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ]), num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    print(f"Shape of prepared training data: {housing_prepared.shape}")
    print("\n")

    # --- 8. Train and Evaluate Models using Cross-Validation ---
    print("--- Training and Evaluating Models ---")
    
    # Linear Regression
    lin_reg = LinearRegression()
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    print("\nLinear Regression Cross-Validation Results:")
    display_scores(lin_rmse_scores)

    # Decision Tree Regressor
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                                  scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)
    print("\nDecision Tree Cross-Validation Results:")
    display_scores(tree_rmse_scores)

    # Random Forest Regressor
    forest_reg = RandomForestRegressor(random_state=42)
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    print("\nRandom Forest Cross-Validation Results:")
    display_scores(forest_rmse_scores)
    print("\n")

    # --- 9. Fine-Tune the Random Forest Model with GridSearchCV ---
    print("--- Fine-Tuning Random Forest with GridSearchCV ---")
    print("This may take a few minutes...")
    param_grid = [
        {'n_estimators': [30, 50, 70], 'max_features': [6, 8, 10]},
        {'bootstrap': [False], 'n_estimators': [10, 20], 'max_features': [3, 4, 5]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best RMSE from CV: ${np.sqrt(-grid_search.best_score_):,.2f}")
    print("\n")

    # --- 10. Final Evaluation on the Test Set ---
    print("--- Final Evaluation on the Test Set ---")
    final_model = grid_search.best_estimator_
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print(f"Final RMSE on Test Set: ${final_rmse:,.2f}")

if __name__ == "__main__":
    main()
