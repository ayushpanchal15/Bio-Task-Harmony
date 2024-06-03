# logistic regression,random forest classification, support vector classifier, mlp

import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from fuzzywuzzy import process

# Loading the dataset
dataset = pd.read_csv('sorted_data.csv')

# Calculating average values for 'Calories (per hour)' column
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

# Sorting tasks based on average calories
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

# Example user input tasks and stored in a list
user_input_tasks = []

# Allowing the user to input tasks
while True:
    task = input("Enter a task (or 'done' to finish): ")
    if task.lower() == 'done':
        break
    user_input_tasks.append(task)

# Calculating average calories for user input tasks
task_averages = calculate_average_calories(dataset, user_input_tasks)

# Calculating calories burnt for user input tasks
calories_burnt = calculate_calories_burnt(user_input_tasks, task_averages)

# Sorting tasks based on calories burnt
prioritized_tasks_calories_burnt = sorted(calories_burnt.items(), key=lambda x: x[1], reverse=True)

# Displaying input tasks and prioritized tasks based on calories burnt
print("\nInput Tasks and Calories Burnt:")
for task, calories in calories_burnt.items():
    print(f"You will burn an average of {calories} while doing this task - {task}")

print("\nPrioritized Tasks based on Calories Burnt:")
for task, _ in prioritized_tasks_calories_burnt:
    print("- " + task)

# Separating the dataset for training the model
X = dataset.drop(['Task', 'Calories (per hour)', 'Wake-up Time', 'Sleep Time'], axis=1)
y = dataset['Calories (per hour)'].apply(lambda x: sum(map(int, x.split('-')))/2)

# Converting the target variable to a binary classification problem
y_class = (y >= y.median()).astype(int)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Defining transformers for numeric and categorical features
numeric_features = ['Working Hours', 'Avg Tasks per Day', 'Age']
categorical_features = ['End of Day Exhausted']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Creating preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating and training different classification models

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', logistic_model)])
logistic_pipeline.fit(X_train, y_train)

# Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', random_forest_model)])
random_forest_pipeline.fit(X_train, y_train)

# Support Vector Classifier
svc_model = SVC(kernel='linear', C=1.0)
svc_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', svc_model)])
svc_pipeline.fit(X_train, y_train)

# Multi-layer Perceptron (Deep Learning)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200)
mlp_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', mlp_model)])
mlp_pipeline.fit(X_train, y_train)

# Making predictions on the test set and evaluating each model
logistic_predictions = logistic_pipeline.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
print(f'Logistic Regression Accuracy: {logistic_accuracy}')

random_forest_predictions = random_forest_pipeline.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
print(f'Random Forest Accuracy: {random_forest_accuracy}')

svc_predictions = svc_pipeline.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_predictions)
print(f'Support Vector Classifier Accuracy: {svc_accuracy}')

mlp_predictions = mlp_pipeline.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
print(f'MLP Accuracy: {mlp_accuracy}')
