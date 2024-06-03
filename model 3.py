# Regularization
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from fuzzywuzzy import process

# Load the dataset
dataset = pd.read_csv('sorted_data.csv')

# Function to calculate average values for 'Calories (per hour)' column
def calculate_average_calories(dataset, tasks):
    task_averages = {}
    for task in tasks:
        if task in dataset['Task'].values:
            task_calories = dataset[dataset['Task'] == task]['Calories (per hour)'].values[0]
            task_average = sum(map(int, task_calories.split('-'))) / 2
            task_averages[task] = task_average
        else:
            matched_task, _ = process.extractOne(task, dataset['Task'].values)
            task_calories = dataset[dataset['Task'] == matched_task]['Calories (per hour)'].values[0]
            task_average = sum(map(int, task_calories.split('-'))) / 2
            task_averages[task] = task_average
    return task_averages

# Function to sort tasks based on average calories
def sort_tasks_by_average_calories(task_averages, user_input_tasks):
    sorted_tasks = []
    for task in user_input_tasks:
        if task in task_averages:
            sorted_tasks.append((task, task_averages[task]))
        else:
            sorted_tasks.append((task, float('-inf')))
    sorted_tasks = sorted(sorted_tasks, key=lambda x: x[1], reverse=True)
    return [task[0] for task in sorted_tasks]

def calculate_calories_burnt(user_input_tasks, task_averages):
    calories_burnt = {}
    for task in user_input_tasks:
        if task in task_averages:
            time = float(input(f"Enter the time for task '{task}' (in hours): "))
            calories = task_averages[task] * time
            calories_burnt[task] = calories
    return calories_burnt

# Example user input tasks
user_input_tasks = []

# Allow the user to input tasks
while True:
    task = input("Enter a task (or 'done' to finish): ")
    if task.lower() == 'done':
        break
    user_input_tasks.append(task)

# Calculate average calories for user input tasks
task_averages = calculate_average_calories(dataset, user_input_tasks)

# Calculate calories burnt for user input tasks
calories_burnt = calculate_calories_burnt(user_input_tasks, task_averages)

# Sort tasks based on calories burnt
prioritized_tasks_calories_burnt = sorted(calories_burnt.items(), key=lambda x: x[1], reverse=True)

# Display input tasks and prioritized tasks based on calories burnt
print("\nInput Tasks and Calories Burnt:")
for task, calories in calories_burnt.items():
    print(f"You will burn an average of {calories} while doing this task - {task}")

print("\nPrioritized Tasks based on Calories Burnt:")
for task, _ in prioritized_tasks_calories_burnt:
    print("- " + task)

# Separate the dataset for training the model
X = dataset.drop(['Task', 'Calories (per hour)', 'Wake-up Time', 'Sleep Time'], axis=1)
y = dataset['Calories (per hour)'].apply(lambda x: sum(map(int, x.split('-')))/2)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define transformers for numeric and categorical features
numeric_features = ['Working Hours', 'Avg Tasks per Day', 'Age']
categorical_features = ['End of Day Exhausted']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and train bagging regressor with regularization
bagging_model = BaggingRegressor(RandomForestRegressor(max_depth=10, min_samples_split=5), n_estimators=10, random_state=42)
bagging_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', bagging_model)])
bagging_pipeline.fit(X_train, y_train)

# Make predictions on the test set
bagging_predictions = bagging_pipeline.predict(X_test)

# Evaluate the bagging model
bagging_mse = mean_squared_error(y_test, bagging_predictions)
print(f'Bagging Mean Squared Error: {bagging_mse}')

# Create and train AdaBoost regressor with regularization
adaboost_model = AdaBoostRegressor(RandomForestRegressor(max_depth=10, min_samples_split=5), n_estimators=50, random_state=42)
adaboost_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', adaboost_model)])
adaboost_pipeline.fit(X_train, y_train)

# Make predictions on the test set
adaboost_predictions = adaboost_pipeline.predict(X_test)

# Evaluate the AdaBoost model
adaboost_mse = mean_squared_error(y_test, adaboost_predictions)
print(f'AdaBoost Mean Squared Error: {adaboost_mse}')
