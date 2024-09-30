import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os


"""
Taken that first all data clean, transfered and 
then again saved into their respective csv s
which we are going to use here 
"""

# constants
# split test data 25%
K = 10
SPLIT = 0.75

# Load data
def load_data(path: str = "/path/to/csv/"):
    """
    This function takes a path string to a CSV file and loads it into
    a Pandas DataFrame.

    :param      path (optional): str, relative path of the CSV file

    :return     df: pd.DataFrame
    """

    df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# Create target variable and predictor variables
def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "estimated_stock_pct"
):
    """
    This function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised 
    machine learning model.

    :param      data: pd.DataFrame, dataframe containing data for the 
                      model
    :param      target: str (optional), target variable that you want to predict

    :return     X: pd.DataFrame
                y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Train algorithm
def train_algorithm_with_cross_validation(
    X: pd.DataFrame = None, 
    y: pd.Series = None,
    K: int = 10
):
    """
    This function takes the predictor and target variables and
    trains a Random Forest Regressor model across K folds. Using
    cross-validation, performance metrics will be output for each
    fold during training.

    :param      X: pd.DataFrame, predictor variables
    :param      y: pd.Series, target variable

    :return
    """

    # Create a list that will store the accuracies of each fold
    accuracy = []

    # Enter a loop to run K folds of cross-validation
    for fold in range(0, K):

        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")
    
    
# load csv and convert into pandas dataframes
# assign directory
directory = 'resources'

# iterate over all clean and converted dataframe into csv files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        if filename == 'sales.csv':
            sales_df = load_data(f)
        elif filename == 'sensor_stock_levels.csv':
            stock_df = load_data(f)
        elif filename == 'sensor_storage_temperature.csv':
            temp_df = load_data(f)
        else:
            print('Not a required csv data file')
            
# agreegate on quantity sum for sales_df after group by product_id and timestamp 
sales_agg = sales_df.groupby(['product_id', 'timestamp']).agg({'quantity':'sum'}).reset_index()

# agreegate on estimated_stock_pct mean for stock_agg group by product_id and timestamp
stock_agg = stock_df.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()
        
# agreegate temperature mean after group by timestamp
temp_agg = temp_df.groupby(['timestamp']).agg({'temperature':'mean'}).reset_index()

# merging sales_agg with stock_agg relate on timestamp and product_id left join
merged_df = stock_agg.merge(sales_agg, on = ['timestamp', 'product_id'], how = 'left')

# merge temp_agg with merge_df relate on timestamp left join 
merged_df = merged_df.merge(temp_agg, on = ['timestamp'], how = 'left')

# fill with 0 where NaN value in quantity col
merged_df['quantity'] = merged_df['quantity'].fillna(0)

# create categories and price col date after drop all duplicate rows
product_categories = sales_df[['product_id', 'category']]
product_categories = product_categories.drop_duplicates()

product_price = sales_df[['product_id', 'unit_price']]
product_price = product_price.drop_duplicates()

# merge product_categories to merged_df
merged_df = merged_df.merge(product_categories, on="product_id", how="left")

# merge product_price with merge_df
merged_df = merged_df.merge(product_price, on="product_id", how="left")

# create day of month, day of week and hour from timestamp col and drop timestamp after
# merged_df['timestamp_day_of_month'] = merged_df['timestamp'].dt.day
merged_df['timestamp_day_of_month'] = pd.DatetimeIndex(merged_df['timestamp']).day
# merged_df['timestamp_day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['timestamp_day_of_week'] = pd.DatetimeIndex(merged_df['timestamp']).weekday
# merged_df['timestamp_hour'] = merged_df['timestamp'].dt.hour
merged_df['timestamp_hour'] = pd.DatetimeIndex(merged_df['timestamp']).hour
merged_df.drop(columns=['timestamp'], inplace=True)

# get dummy col from categories data so that each cateogry relate to that row 0 or 1 based on if that category present in the row
merged_df = pd.get_dummies(merged_df, columns = ['category'])

# product_id col not required
merged_df.drop(columns = ['product_id'], inplace=True)

# seperating x and y for model fit data
X, y = create_target_and_predictors(merged_df, 'estimated_stock_pct')

# train the algorithm, model
train_algorithm_with_cross_validation(X, y, K)