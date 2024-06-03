import 'dart:convert';
import 'dart:html';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(TaskPrioritizationApp());
}

class TaskPrioritizationApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Task Prioritization',
      theme: ThemeData(
        primaryColor: Color(0xFF1f2029),
        scaffoldBackgroundColor: Color(0xFF1f2029),
        fontFamily: 'Poppins',
        colorScheme: ColorScheme.fromSwatch(
          accentColor: Color(0xFFFFEBA7),
        ).copyWith(
          secondary:
              Color(0xFFFFEBA7), // This is also required for some widgets
        ),
      ),
      home: TaskPrioritizer(),
    );
  }
}

class TaskPrioritizer extends StatefulWidget {
  @override
  _TaskPrioritizerState createState() => _TaskPrioritizerState();
}

class _TaskPrioritizerState extends State<TaskPrioritizer> {
  TextEditingController _tasksController = TextEditingController();
  TextEditingController _hoursController = TextEditingController();
  List<String> _prioritizedTasks = [];

  @override
  void dispose() {
    _tasksController.dispose();
    _hoursController.dispose();
    super.dispose();
  }

  Future<void> _prioritizeTasks() async {
    final tasks = _tasksController.text.split(',');
    final hours = _hoursController.text.split(',');

    if (tasks.length != hours.length) {
      // Show an error message or handle the mismatched input
      return;
    }

    final response = await http.get(Uri.parse('dataset.json'));
    final dataset = json.decode(response.body);

    final prioritizedTasks = List.generate(tasks.length, (index) {
      final trimmedTask = tasks[index].trim().toLowerCase();
      final taskKey = _findTaskKey(trimmedTask, dataset);
      final calories = (dataset[taskKey] as num?)?.toDouble() ??
          0 * double.parse(hours[index].trim());
      return MapEntry(trimmedTask, calories);
    });

    prioritizedTasks.sort((a, b) => b.value.compareTo(a.value));

    setState(() {
      _prioritizedTasks = prioritizedTasks.map((entry) => entry.key).toList();
    });
  }

  String _findTaskKey(String task, Map dataset) {
    if (dataset.containsKey(task)) {
      return task;
    }

    for (final key in dataset.keys) {
      if ((dataset[key] as List).contains(task)) {
        return key;
      }
    }

    // Return a default value if the task key is not found
    return '';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Task Prioritization'),
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Enter tasks (separated by commas):',
                style: TextStyle(color: Colors.white),
              ),
              TextField(
                controller: _tasksController,
                style: TextStyle(color: Colors.white),
                decoration: InputDecoration(
                  filled: true,
                  fillColor: Color(0xFF1f2029),
                  border: OutlineInputBorder(),
                  hintText: 'Enter tasks...',
                  hintStyle: TextStyle(color: Colors.white),
                ),
              ),
              SizedBox(height: 20.0),
              Text(
                'Enter hours for tasks (separated by commas):',
                style: TextStyle(color: Colors.white),
              ),
              TextField(
                controller: _hoursController,
                style: TextStyle(color: Colors.white),
                decoration: InputDecoration(
                  filled: true,
                  fillColor: Color(0xFF1f2029),
                  border: OutlineInputBorder(),
                  hintText: 'Enter hours...',
                  hintStyle: TextStyle(color: Colors.white),
                ),
              ),
              SizedBox(height: 20.0),
              ElevatedButton(
                onPressed: _prioritizeTasks,
                child: Text('Prioritize Tasks'),
              ),
              SizedBox(height: 20.0),
              Text(
                'Output:',
                style: TextStyle(color: Colors.white),
              ),
              SizedBox(height: 10.0),
              ListView.builder(
                shrinkWrap: true,
                itemCount: _prioritizedTasks.length,
                itemBuilder: (context, index) {
                  return Card(
                    color: Color(0xFF1f2029),
                    margin: EdgeInsets.symmetric(vertical: 5.0),
                    child: ListTile(
                      title: Text(
                        _prioritizedTasks[index],
                        style: TextStyle(color: Colors.white),
                      ),
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}


// the above code is for prioritization page


// ------------------------------------------------------------------------------------------------------------------------


import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';

void main() {
  runApp(LandingPage());
}

class LandingPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Harmony',
      theme: ThemeData(
        scaffoldBackgroundColor: Color(0xFF1F2029),
        textTheme: TextTheme(
          bodyText1: TextStyle(color: Colors.white),
        ),
      ),
      home: Scaffold(
        body: SingleChildScrollView(
          child: Column(
            children: [
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 50, vertical: 10),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Container(
                      height: 100,
                      child: Image.asset('assets/harmony-logo.png'),
                    ),
                    // You can add any other navigation items here
                  ],
                ),
              ),
              SizedBox(height: 100),
              Row(
                children: [
                  SizedBox(width: 50),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'A personalized application solution\n to the everday struggles of tasks management.',
                        style: TextStyle(fontSize: 32),
                      ),
                      SizedBox(height: 20),
                      ElevatedButton(
                        onPressed: () {
                          // Add your navigation logic here
                        },
                        child: Text('Enter'),
                      ),
                    ],
                  ),
                  SizedBox(width: 50),
                  Expanded(
                    child: Image.asset('assets/landing_page.png'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}


// the aboe code is for landing page


// -------------------------------------------------------------------------------------------------------------------

