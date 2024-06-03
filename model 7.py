import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from fuzzywuzzy import process
from scipy.stats import uniform, randint

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
print()
print()

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

# Define Random Forest Classifier
random_forest_classifier = RandomForestClassifier(random_state=42)

# Create pipeline
random_forest_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', random_forest_classifier)])

# Define parameter distribution for Randomized Search
param_dist = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': [None, randint(5, 30)],
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'model__max_features': ['auto', 'sqrt', uniform(0.1, 1.0)],
    'model__bootstrap': [True, False],
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(random_forest_pipeline, param_dist, cv=5, scoring='accuracy', verbose=2, n_iter=100, random_state=42)

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)

# Print the best parameters and the corresponding accuracy
print(f'Best parameters: {random_search.best_params_}')
print(f'Best accuracy: {random_search.best_score_}')

# Refit the model on the entire training data using the best parameters
random_forest_pipeline.set_params(**random_search.best_params_)
random_forest_pipeline.fit(X_train, y_train)

# Make predictions on the test set and evaluate the model
random_forest_predictions = random_forest_pipeline.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
print(f'Random Forest Classifier Accuracy: {random_forest_accuracy}')