from flask import Flask, render_template, request, jsonify
import pandas as pd
from fuzzywuzzy import process

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('sorted_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prioritize', methods=['POST'])
def prioritize_tasks():
    try:
        # Get input values from the request
        tasks_input = request.form['tasks']
        times_input = request.form['times']

        # Convert input strings to arrays
        user_input_tasks = tasks_input.split(',')
        user_input_times = times_input.split(',')

        # Call Python script to prioritize tasks (replace this with your actual logic)
        task_averages = calculate_average_calories(dataset, user_input_tasks)
        calories_burnt = calculate_calories_burnt(user_input_tasks, task_averages, user_input_times)
        prioritized_tasks_calories_burnt = sorted(calories_burnt.items(), key=lambda x: x[1], reverse=True)
        prioritized_tasks = [task[0] for task in prioritized_tasks_calories_burnt]

        # Return the results as JSON
        return jsonify({
            'inputTasks': user_input_tasks,
            'inputTimes': user_input_times,
            'caloriesBurnt': calories_burnt,
            'prioritizedTasks': prioritized_tasks
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

def calculate_calories_burnt(user_input_tasks, task_averages, user_input_times):
    calories_burnt = {}
    for task, time in zip(user_input_tasks, user_input_times):
        if task in task_averages:
            time = float(time)
            calories = task_averages[task] * time
            calories_burnt[task] = calories
    return calories_burnt

if __name__ == '__main__':
    app.run(debug=True)
