# RandomForest classifier and SVC. Implemented model stacking with randomforest as base model and svc as final estimator. accuracy : 0.64
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from fuzzywuzzy import process

# Loading the dataset
dataset = pd.read_csv('sorted_data.csv')

# This is a function to calculate average values for 'Calories (per hour)' column
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

# This is a function for custom feature engineering
def custom_feature_engineering(X):
    # Add custom features here
    return X

# Appending custom feature engineering step to the preprocessor
preprocessor = Pipeline(steps=[
    ('feature_engineering', FunctionTransformer(custom_feature_engineering)),
    ('preprocessor', preprocessor)
])

# Creating base Random Forest model
base_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Creating stacking classifier with Random Forest as the base model
stacking_classifier = StackingClassifier(
    estimators=[('rf', base_rf_model)],
    final_estimator=SVC(random_state=42, probability=True),  # Use SVC as the final estimator
    cv=5
)

# Fitting stacking classifier on training data
stacking_classifier.fit(X_train, y_train)

# Evaluating the stacking classifier
stacking_accuracy = accuracy_score(y_test, stacking_classifier.predict(X_test))

# This is a function to sort tasks based on calories burnt
def sort_tasks(task_averages, user_input_tasks):
    sorted_tasks = []
    for task in user_input_tasks:
        if task in task_averages:
            sorted_tasks.append((task, task_averages[task]))
        else:
            sorted_tasks.append((task, float('-inf')))
    sorted_tasks = sorted(sorted_tasks, key=lambda x: x[1], reverse=True)
    return [task[0] for task in sorted_tasks]

# This is a function to calculate calories burnt for user input tasks
def calculate_calories_burnt(user_input_tasks, task_averages, user_input_times):
    calories_burnt = {}
    for task, time in zip(user_input_tasks, user_input_times):
        if task in task_averages:
            time = float(time)
            calories = task_averages[task] * time
            calories_burnt[task] = calories
    return calories_burnt

# Taking input tasks and times
user_input_tasks = input("Enter tasks (separated by space or comma):\n").split()
user_input_times = input("Enter times for tasks (separated by space or comma):\n").split()

# Stop taking input tasks as soon as 'done' is encountered
while 'done' not in user_input_tasks:
    user_input_tasks.extend(input().split())

# Removing the trailing 'done' from the list of tasks
user_input_tasks.remove('done')

# Calculating average calories for user input tasks
task_averages = calculate_average_calories(dataset, user_input_tasks)

# Sorting tasks based on calories burnt
prioritized_tasks_calories_burnt = sort_tasks(task_averages, user_input_tasks)

# Displaying input tasks and times
print("\nInput Tasks and Times:")
for task, time in zip(user_input_tasks, user_input_times):
    print(f"Task: {task}, Time: {time} hours")

# Calculating calories burnt for user input tasks
calories_burnt = calculate_calories_burnt(user_input_tasks, task_averages, user_input_times)

# Displaying calories burnt for each task
print("\nCalories Burnt for Each Task:")
for task, calories in calories_burnt.items():
    print(f"You will burn an average of {calories} calories while doing task: {task}")

# Displaying input tasks and prioritized tasks based on calories burnt
print("\nPrioritized Tasks based on Calories Burnt:")
for task in prioritized_tasks_calories_burnt:
    print("- " + task)

# Displaying accuracy of the stacking classifier
print(f'\nStacking Classifier Accuracy: {stacking_accuracy}')
