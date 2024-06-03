// Hardcoded tasks and their calories per hour
const tasks = {
    "Running a marathon" : 1500,
    "Heavy construction work" :2250,
    "Firefighting" :2250,
    "Rock climbing" :2150,
    "Swimming long distances" :2100,
    "Scuba diving" :2050,
    "Aerobics" :1850,
    "Cycling long distances" :1650,
    "Dancing" :1550,
    "Mountain biking" :1150,
    "Hiking" :800,
    "Kickboxing" :525,
    "Circuit training" :450,
    "Jumping rope" :251,
    "CrossFit" :500,
    "High-intensity cardio" :650,
    "Rowing" :250,
    "Skateboarding" :250,
    "Pilates" :350,
    "Jogging" :300,
    "Gardening" :180.5,
    "Yoga" : 155,
    "Walking" : 160,
    "Weightlifting" :375,
    "Tennis" :225,
    "Surfing" :200,
    "Martial arts" :125, 
    "Tai Chi" :125,
    "Badminton" :125,
    "Playing musical instrument" :150,
    "Watching TV" :90,
    "Playing basketball" :90,
    "Playing video games" :67.5,
    "Reading" :65,
    "Playing football" :130,
    "Writing" :62.5,
    "Studying" :75,
    "Volleyball" :90,
    "Coding" :62.5,
    "Drawing" :57.5,
    "Playing chess" :52.5,
    "Table tennis" :75,
    "Using a computer" :57.5,
    "Photography" :35,
    "Bird watching" :47.5,
    "Golf" :55,
    "Cooking" :45,
    "Sketching" :35,
    "Origami" :32.5,
    "Cleaning" :250,
    "Calligraphy" :27.5,
    "Bowling" :37.5,
    "Knitting" :27.5,
    "Embroidery" :22.5,
    "Stretching" :17.5,
    "Laundry" :25,
    "Skiing" :25,
    "Vacuuming" :17.5,
    "Snowboarding" :22.5,
    "Pottery" :17.5,
    "Candle making" :125,
    "Ice skating" :425,
    "Dusting" :90,
    "Grocery shopping" :150,
    "Mopping" :200,
    "Hula hooping" :182.5,
    "Jumping jacks" : 900,
    "Trampoline jumping" :400,
    "Juggling" :150
};

function prioritizeTasks() {
    const taskInput = document.getElementById('taskInput').value;
    const hoursInput = document.getElementById('hoursInput').value;

    const userTasks = taskInput.split(/[, ]/).map(task => task.trim().toLowerCase());
    const userHours = hoursInput.split(/[, ]/).map(hour => parseFloat(hour.trim()));

    // Validate inputs
    if (userTasks.length !== userHours.length) {
        alert("Number of tasks and hours must be the same.");
        return;
    }

    // Create a map for user tasks and their corresponding calories
    const userTaskCalories = {};
    userTasks.forEach((task, index) => {
        // Check if the task contains any hardcoded task keyword
        const matchingTask = Object.keys(tasks).find(hardcodedTask =>
            task.includes(hardcodedTask)
        );

        if (matchingTask) {
            const calories = tasks[matchingTask] * userHours[index];
            userTaskCalories[task] = calories;
        }
    });

    // Prioritize tasks based on calories burnt
    const prioritizedTasks = Object.entries(userTaskCalories)
        .sort((a, b) => b[1] - a[1])
        .map(task => task[0]);

    // Display the prioritized tasks
    const taskList = document.getElementById('taskList');
    taskList.innerHTML = "";
    prioritizedTasks.forEach(task => {
        const listItem = document.createElement('li');
        listItem.textContent = task;
        taskList.appendChild(listItem);
    });
}
