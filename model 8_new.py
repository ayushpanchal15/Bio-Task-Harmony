# Random Forest Classification
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# Separate the dataset for training the model
X = dataset.drop(['Task', 'Calories (per hour)', 'Wake-up Time', 'Sleep Time'], axis=1)
y = dataset['Calories (per hour)'].apply(lambda x: sum(map(int, x.split('-')))/2)

# Convert the target variable to a binary classification problem
y_class = (y >= y.median()).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

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

# Create and train Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', random_forest_model)])
random_forest_pipeline.fit(X_train, y_train)


def sort_tasks(task_averages, user_input_tasks):
    sorted_tasks = []
    for task in user_input_tasks:
        if task in task_averages:
            sorted_tasks.append((task, task_averages[task]))
        else:
            sorted_tasks.append((task, float('-inf')))
    sorted_tasks = sorted(sorted_tasks, key=lambda x: x[1], reverse=True)
    return [task[0] for task in sorted_tasks]

def calculate_calories_burnt(user_input_tasks, task_averages, user_input_times):
    calories_burnt = {}
    for task, time in zip(user_input_tasks, user_input_times):
        if task in task_averages:
            time = float(time)
            calories = task_averages[task] * time
            calories_burnt[task] = calories
    return calories_burnt

# Example user input tasks
user_input_tasks = input("Enter tasks (separated by space or comma):\n").split()

# Example user input times
user_input_times = input("Enter times for tasks (separated by space or comma):\n").split()

# Stop taking input tasks as soon as 'done' is encountered
while 'done' not in user_input_tasks:
    user_input_tasks.extend(input().split())

# Remove the trailing 'done' from the list of tasks
user_input_tasks.remove('done')

# Calculate average calories for user input tasks
task_averages = calculate_average_calories(dataset, user_input_tasks)

# Display input tasks and times
print("\nInput Tasks and Times:")
for task, time in zip(user_input_tasks, user_input_times):
    print(f"Task: {task}, Time: {time} hours")

# Calculate calories burnt for user input tasks
calories_burnt = calculate_calories_burnt(user_input_tasks, task_averages, user_input_times)

# Display calories burnt for each task
print("\nCalories Burnt for Each Task:")
for task, calories in calories_burnt.items():
    print(f"You will burn an average of {calories} calories while doing task: {task}")

# Sort tasks based on calories burnt
prioritized_tasks_calories_burnt = sorted(calories_burnt.items(), key=lambda x: x[1], reverse=True)

# Display input tasks and prioritized tasks based on calories burnt
print("\nPrioritized Tasks based on Calories Burnt:")
for task, _ in prioritized_tasks_calories_burnt:
    print("- " + task)


# Make predictions on the test set and evaluate the model
random_forest_predictions = random_forest_pipeline.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
print(f'\nRandom Forest Classifier Accuracy: {random_forest_accuracy}')

"""
import joblib

# Save the trained model to a file
joblib.dump(random_forest_pipeline, 'random_forest_model.joblib')"""
