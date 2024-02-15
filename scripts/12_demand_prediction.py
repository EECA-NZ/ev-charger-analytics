import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import ElasticNetCV
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures


# Constants


# Functions

def preprocess_data(data, excluded_columns, target=None, impute=True):
    """
    Prepares the dataset for training the model.

    Parameters:
    - data (DataFrame): The dataset to be processed.
    - excluded_columns (list of str): The columns to be excluded from the dataset.
    - impute (bool): Determines whether to impute missing values or drop rows with missing values.

    Returns:
    - X (DataFrame): The processed feature matrix.
    - y (Series): The target variable.
    """
    if target is None:
        raise ValueError("Target variable must be specified.")
    data_copy = data.copy()
    X = data_copy[[x for x in data_copy.columns if x not in excluded_columns]]
    y = data_copy[target]
    # Drop rows where target variable is NaN
    non_nan_rows = ~y.isna()
    X = X[non_nan_rows]
    y = y[non_nan_rows]
    # Drop columns that are entirely NaN
    X = X.dropna(axis=1, how='all')
    # Encode categorical features
    le = LabelEncoder()
    for column in X.columns:
        if X[column].dtype == 'object' or X[column].dtype == 'bool':
            X[column] = le.fit_transform(X[column])
    if impute:
        # Impute NaN values in features
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    else:
        # Drop rows with any NaN values in X and the corresponding rows in y
        non_nan_rows = ~X.isna().any(axis=1)
        X = X[non_nan_rows]
        y = y[non_nan_rows]
    return X, y


def plot_predictions(y_test, y_pred, title, filename):
    """
    Generates and saves a scatter plot of the predicted vs actual values.

    Parameters:
    - y_test (array): Actual target values.
    - y_pred (array): Predicted target values from the model.
    - title (str): The title of the plot.
    - filename (str): The path to save the plot image.
    """
    # Set the minimum and maximum bounds for the plot axes
    data_min = 0
    data_max = max(np.ceil(np.max(y_pred)), np.ceil(np.max(y_test)))
    axes_limit = [data_min, data_max]

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_pred, y_test, alpha=0.5)
    plt.plot(axes_limit, axes_limit, 'k--', lw=2)  # 45-degree line for reference
    plt.xlim(axes_limit)
    plt.ylim(axes_limit)
    plt.xlabel('Predicted daily demand')
    plt.ylabel('Actual daily demand')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio square
    plt.savefig(filename)



def print_top_predictors(ridge_model, feature_names, n_top=10):
    coef = ridge_model.coef_
    top_predictors = np.argsort(np.abs(coef))[-n_top:]
    print("Top predictors:")
    for idx in top_predictors:
        print(f"{feature_names[idx]}: {coef[idx]}")


def train_and_evaluate(X, y, model, verbose=True):
    """
    Trains and evaluates the model using the provided data.

    Parameters:
    - X (DataFrame): Feature matrix.
    - y (Series): Target variable.
    - model: The regression model to be used.
    - verbose (bool): If True, prints the model performance metrics.

    Returns:
    - model: The trained regression model.
    - X_test (DataFrame): The feature matrix for the test set.
    - y_test (Series): The target variable for the test set.
    - y_pred (array): The predicted values for the test set.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Fit the model
    model.fit(X_train, y_train)
    # Evaluate
    y_pred = model.predict(X_test)
    correlation_coef, _ = pearsonr(y_pred, y_test)
    r_squared = model.score(X_test, y_test)
    if verbose:
        print(f'Out-of-sample correlation coefficient: {correlation_coef}')
        print(f'Out-of-sample R^2: {r_squared}')
    return model, X_test, y_test, y_pred


def print_top_predictors(model, feature_names, n_top=10):
    coef = model.coef_
    top_indices = np.argsort(np.abs(coef))[-n_top:]
    top_indices = top_indices[coef[top_indices] != 0]  # Exclude zero coefficients
    print("Top predictors:")
    for idx in top_indices:
        print(f"{feature_names[idx]}: {coef[idx]}")


def plot_histogram(data, column_name, filename):
    """
    Plots a histogram for a specified column in the dataset.

    Args:
    - data (DataFrame): The dataset containing the column.
    - column_name (str): The name of the column to plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column_name], kde=True)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.savefig(filename)


def display_summary_statistics(data, column_name):
    """
    Displays summary statistics for a specified column in the dataset.

    Args:
    - data (DataFrame): The dataset containing the column.
    - column_name (str): The name of the column for which to display statistics.
    """
    print(f"Summary statistics for {column_name}:\n")
    print(data[column_name].describe())




# Main

# Load data
turn_in_rates_data = pd.read_csv("../data/turn_in_rates_data.csv")
turn_in_rates_data = turn_in_rates_data[turn_in_rates_data['EV_turnin_rate'] <= 1] # remove outliers

# Define the subsets of data
destination = turn_in_rates_data[turn_in_rates_data.KwRated < 50]
journey = turn_in_rates_data[turn_in_rates_data.KwRated >= 50]

# Columns to exclude
excluded_columns = ['avg_misses_per_day', 'O', 'avg_observed_plugins_per_day',
                    'daily_demand', 'turnin_rate', 'EV_turnin_rate'] + \
                   ['Unnamed: 0', 'SiteId', 'SiteId.1', 'Address', 'Days_observed', 'SA2_name',
                    'Operator.1', 'Images', 'AssetId', 'SiteId', 'Name', 'ChargingStationId',
                    'Chargers_needed', 'ProviderDeleted', 'ProviderDeleted.1',
                    'SA2_code', 'Population_growth', 'geometry'] + \
                    ['Manufacturer', 'Model']

scaler = StandardScaler()


# Train and evaluate models

ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 200), cv=10)
lasso_model = LassoCV(alphas=np.logspace(-6, 6, 200), cv=10, random_state=42, max_iter=10000)
elastic_net_model = ElasticNetCV(alphas=np.logspace(-6, 6, 200), cv=10, random_state=42, max_iter=10000)


print("\nDestination Analysis")
X, y = preprocess_data(destination, excluded_columns, target='daily_demand', impute=True)
X_scaled = scaler.fit_transform(X)
model, X_test, y_test, y_pred = train_and_evaluate(X_scaled, y, ridge_model)
plot_predictions(y_test, y_pred, "Destination Analysis", "../png/destination_analysis.png")
print_top_predictors(model, X.columns)


print("\nJourney Analysis")
X, y = preprocess_data(journey, excluded_columns, target='daily_demand', impute=True)
X_scaled = scaler.fit_transform(X)
model, X_test, y_test, y_pred = train_and_evaluate(X_scaled, y, ridge_model)
plot_predictions(y_test, y_pred, "Journey Analysis", "../png/journey_analysis.png")
print_top_predictors(model, X.columns)


print("\nUnstratified Analysis")
X, y = preprocess_data(turn_in_rates_data, excluded_columns, target='daily_demand', impute=True)
X_scaled = scaler.fit_transform(X)
model, X_test, y_test, y_pred = train_and_evaluate(X_scaled, y, ridge_model)
plot_predictions(y_test, y_pred, "Unstratified Analysis", "../png/unstratified_analysis.png")
print_top_predictors(model, X.columns)


print("\nPlot histogram and display summary statistics for EV_turnin_rate")
plot_histogram(turn_in_rates_data, 'EV_turnin_rate', "../png/ev_turnin_rate.png")
display_summary_statistics(turn_in_rates_data, 'EV_turnin_rate')