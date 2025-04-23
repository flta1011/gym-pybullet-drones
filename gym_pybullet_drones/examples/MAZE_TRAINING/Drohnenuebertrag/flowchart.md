---
config:
  layout: fixed
---
flowchart TD
    Main["main.py"] -- Initializes --> DC["DroneController"]
    Main -- Creates and Displays --> MW["MainWindow"]
    DC -- Establishes Connection --> Connect["Connect to Drone"]
    DC -- Retrieves Data --> GetObs["Get Observations"]
    DC -- Uses AI Model --> PredictAct["Predict Actions"]
    DC -- Sends Commands --> ExecAct["Execute Actions"]
    DC -- Ensures Safety and Disconnects --> Disconnect["Safety Features and Disconnect"]
    DC -- Manages Observations --> OM["ObsManager"]
    OM -- Updates Map --> SLAM["SLAM Map"]
    OM -- Generates --> ObsSpace["Observation Space"]
    MW -- Displays --> Controls["Control Buttons"]
    MW -- Shows --> Status["Status Display"]
    MW -- Visualizes --> SLAM
    Controls -- Sends User Input --> DC
    Connect -- Once Connected --> GetObs
    GetObs -- Feeds Data to --> PredictAct
    PredictAct -- Determines Actions --> ExecAct
    ExecAct -- Loops Back for --> GetObs
