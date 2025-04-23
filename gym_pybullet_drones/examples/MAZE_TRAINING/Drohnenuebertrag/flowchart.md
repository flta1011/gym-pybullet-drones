---
config:
  themeVariables:
    fontFamily: Arial
    fontSize: 16px
  theme: forest
  layout: fixed
---
flowchart TD
 subgraph Configuration["Configuration"]
        ObsType["Observation Type"]
        ActType["Action Type"]
        ModelType["Model Type"]
        ModelPath["ML Model Path"]
  end
    Main["main.py"] -- Creates --> DC["DroneController"] & MW["MainWindow"]
    DC -- Configures --> Configuration
    DC -- Loads --> ModelPath
    DC -- Manages --> Connect["Connect to Drone"]
    DC -- Retrieves --> GetObs["Get Observations"]
    DC -- Uses model for --> PredictAct["Predict Actions"]
    DC -- Sends commands to --> ExecAct["Execute Actions"]
    DC -- Handles --> Disconnect["Safety Features and Disconnect from Drone"]
    PredictAct -- AI Decision --> AIControl["AI Control Logic"]
    AIControl -- Updates --> ExecAct
    MW -- Displays --> Controls["Control Buttons"]
    MW -- Shows --> Status["Status Display"] & SLAM["SLAM Map Display"]
    Controls -- User Input --> DC
    DC -- Updates --> Status
    DC -- Provides --> SLAM
    DC -- SLAM Data --> SLAM
    Connect -- Once connected --> GetObs
    GetObs -- Feeds data to --> PredictAct
    PredictAct -- Determines --> ExecAct
    ExecAct -- Loop back for --> GetObs
    Connect -- Connection Failed --> Error["Error Handling"]
     ObsType:::flow
     ActType:::flow
     ModelType:::flow
     ModelPath:::flow
     Main:::main
     DC:::controller
     MW:::gui
     Connect:::controller
     GetObs:::controller
     PredictAct:::controller
     ExecAct:::controller
     Disconnect:::controller
     AIControl:::controller
     Controls:::gui
     Status:::gui
     SLAM:::gui
    classDef main fill:#fce4ff,stroke:#333,stroke-width:2px,color:#333
    classDef controller fill:#e6e6ff,stroke:#333,stroke-width:1px,color:#333
    classDef gui fill:#e6ffe6,stroke:#333,stroke-width:1px,color:#333
    classDef flow fill:#fff8e6,stroke:#333,stroke-width:1px,color:#333,stroke-dasharray: 5 5
