# Randomforest classifier and SVC, implementing weighted voting
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
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

# Function for custom feature engineering
def custom_feature_engineering(X):
    # Add custom features here
    return X

# Append custom feature engineering step to the preprocessor
preprocessor = Pipeline(steps=[
    ('feature_engineering', FunctionTransformer(custom_feature_engineering)),
    ('preprocessor', preprocessor)
])

# Create and train RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', random_forest_model)])

# Calculate cross-validation scores for RandomForestClassifier
rf_cv_scores = cross_val_score(random_forest_pipeline, X_train, y_train, cv=5)

# Create and train SVC
from sklearn.svm import SVC

svc_model = SVC(random_state=42, probability=True)  # Enable probability estimates for soft voting
svc_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', svc_model)])

# Calculate cross-validation scores for SVC
svc_cv_scores = cross_val_score(svc_pipeline, X_train, y_train, cv=5)

# Calculate weights for weighted voting
rf_weight = rf_cv_scores.mean()
svc_weight = svc_cv_scores.mean()
total_weight = rf_weight + svc_weight
rf_weight /= total_weight
svc_weight /= total_weight         

# Fit the ensemble model with weighted voting
ensemble_model = VotingClassifier(estimators=[
    ('random_forest', random_forest_pipeline),
    ('svc', svc_pipeline)
], voting='soft', weights=[rf_weight, svc_weight])

ensemble_model.fit(X_train, y_train)

# Evaluate the ensemble model
ensemble_accuracy = accuracy_score(y_test, ensemble_model.predict(X_test))

# Function to calculate calories burnt for user input tasks
def calculate_calories_burnt(user_input_tasks, task_averages, user_input_times):
    calories_burnt = {}
    for task, time in zip(user_input_tasks, user_input_times):
        if task in task_averages:
            time = float(time)
            calories = task_averages[task] * time
            calories_burnt[task] = calories
    return calories_burnt

# Example user input tasks and times
user_input_tasks = input("Enter tasks (separated by space or comma):\n").split()
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

# Sort tasks based on calories burnt
prioritized_tasks_calories_burnt = sorted(calories_burnt.items(), key=lambda x: x[1], reverse=True)

# Display input tasks and prioritized tasks based on calories burnt
print("\nPrioritized Tasks based on Calories Burnt:")
for task, _ in prioritized_tasks_calories_burnt:
    print("- " + task)

# Display accuracy of the ensemble model
print(f'\nEnsemble Model Accuracy with Weighted Voting: {ensemble_accuracy}')

