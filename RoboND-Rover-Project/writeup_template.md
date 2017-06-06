## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

The first task is to identify the environment i.e walls, rocks and ground. I have followed the principles as mentioned in the lesson to do filtering based on RGB values. These were based on trial and error to identify , the iterations resulted in solution.py

Used this to fill the contents of the function perception_step in perception.py with the changes for Rover.pos instead of the data xpos and ypos and also Rover.yaw.
To identify the rock or wall is based in the polar co-ordinate space analysis.
Using this the Rover properties are updated to move forward or stop.

The decision_step in decision.py uses the current state of the Rover and updates the movement parameters of the Robot as defined in the decision.py.
I am re-working on this step to make the decision making smoother. Currently this is my logic
