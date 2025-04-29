# Projektzusammenfassung

Dieses Projekt untersucht den Einsatz von Deep Reinforcement Learning (DRL), um eine Bitcraze Crazyflie-Drohne für die autonome Navigation in einer Labyrinth-Umgebung zu trainieren.

Die Drohne ist mit minimaler Sensortechnik ausgestattet:

- Multi-Ranger Deck (ToF-Sensoren)

- Flow Deck v2

- Crazyradio 2.0

## Wichtige Punkte

### Trainingsumgebung

- Simulation: gym-pybullet-drones auf Basis von PyBullet und OpenAI Gym-Standards.

- Features: Realistische Drohnenphysik und Sensorsimulation.

### Ziel

- Die Drohne soll in einem komplexen Labyrinth ein unbekanntes Ziel möglichst schnell erreichen (Drone-Race-Szenario).

- Gesteuert wird die Geschwindigkeit über trainierte DRL-Policies.

### Verwendete DRL-Modelle

- PPO (Proximal Policy Optimization)

- SAC (Soft Actor-Critic)

- DQN (Deep Q-Network) mit einem SLAM-basierten Ansatz:

    - Einbindung von Karteninformationen über CNN-Policy oder Multi-Input-Policy.

### Trainingsansatz

- Schrittweise Komplexitätssteigerung der Navigationsaufgaben.

- Fokus auf Optimierung von Reward-Funktionen und Hyperparametern zur Performancesteigerung.

### Simulationsumgebung

- Labyrinthgröße: ca. 3×3 Meter

- Besonderheiten:

    - Mindestens 30 cm breite Passagen

    - Enge Durchgänge

    - Sackgassen und verwinkelte Wege

### Übertragung auf reale Drohne

- Nach dem Training wurden die Modelle auf die reale Crazyflie-Drohne übertragen.

- #### Ergebnisse:

    - Deutlich schlechtere Leistung als in der Simulation.

    - Geringere Konstanz beim Abstandhalten zu Wänden.


# Beispiele zur Überprüfung der Installation


This is a minimalist refactoring of the original `gym-pybullet-drones` repository, designed for compatibility with [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium), [`stable-baselines3` 2.0](https://github.com/DLR-RM/stable-baselines3/pull/1327), and SITL [`betaflight`](https://github.com/betaflight/betaflight)/[`crazyflie-firmware`](https://github.com/bitcraze/crazyflie-firmware/).


## Installation um dieses Projekt anwenden zu könnens

Tested on Intel x64/Ubuntu 22.04 and Apple Silicon/macOS 14.1.

```sh
git clone https://github.com/flta1011/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`


```


## This are examples so check if the installation was succesfull

### PID control examples

```sh
cd gym_pybullet_drones/examples/
python3 pid.py # position and velocity reference
python3 pid_velocity.py # desired velocity reference
```

### Downwash effect example

```sh
cd gym_pybullet_drones/examples/
python3 downwash.py
```

### Reinforcement learning examples (SB3's PPO)

```sh
cd gym_pybullet_drones/examples/
python learn.py # task: single drone hover at z == 1.0
python learn.py --multiagent true # task: 2-drone hover at z == 1.2 and 0.7
```

<img src="gym_pybullet_drones/assets/rl.gif" alt="rl example" width="375"> <img src="gym_pybullet_drones/assets/marl.gif" alt="marl example" width="375">

### Installation die bei unserer Projektumsetzung hinzukamen

Es kann zu Problemen von nicht installierten Bibliotheken kommen. Diese entsprechend Nachinstallieren. Vier Beispiele hierfür sind angegeben

```sh
pip install pyyaml
pip install pandas as pd 
pip install dash
pip install stable-baselines3[extra]


```

# Projektstruktur

📂 /assets/
    └─ URDF-Dateien für Physik-Engine und Simulation (Labyrinthe, Hindernisse)

📂 /control/
    └─ Steuerungsklassen und Basisbeispiele für Drohnenkontrolle

📂 /envs/
    └─ Alte Beispielumgebungen (Legacy-Code)

📂 /utils/
    └─ Hilfsklassen und Simulationsmodule

📂 /Auswertung_der_Modelle/
    └─ Logging und Auswertung der Trainingsdaten

📂 /examples/
    ├─ Hauptarbeitsordner für Trainings-, Simulations- und Übertragungsskripte
    ├─ /MAZE_TRAINING/
        ├─ Training in Labyrinthen und Modellübertragung auf reale Drohne
        ├─ Wichtige Dateien:
            - learn_MAZE_TRAINING.py: Training/Test Konfiguration & Logging
            - Maze_init_position_checker_visual.py: Startpositionsvalidierung
            - BaseAviary_MAZE_TRAINING.py: Definiert Action/Observation Space, Rewards, Abbruchkriterien
            - SimpleSlam_MAZE_TRAINING.py: SLAM-Map Erstellung und Aktualisierung

📂 /Drohnenuebertrag/
    └─ Übertragungsskripte (DQN/SAC) für reale Drohne (verschiedene Observation Spaces)

📂 /Drohnenuebertrag_2/
    └─ Finalisierte, stabilere Version für reale Drohneneinsätze

📂 /examples/maze_urdf_test/self_made_maps/
    └─ Maze Generator:
        - maze_generator.py: Erzeugt URDF-Dateien aus .csv Kartendateien

📂 /maps/
    └─ Enthält die Kartendateien (.csv) für Labyrinth-Generierung



