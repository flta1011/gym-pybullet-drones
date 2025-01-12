# gym-pybullet-drones

This is a minimalist refactoring of the original `gym-pybullet-drones` repository, designed for compatibility with [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium), [`stable-baselines3` 2.0](https://github.com/DLR-RM/stable-baselines3/pull/1327), and SITL [`betaflight`](https://github.com/betaflight/betaflight)/[`crazyflie-firmware`](https://github.com/bitcraze/crazyflie-firmware/).

> **NOTE**: if you prefer to access the original codebase, presented at IROS in 2021, please `git checkout [paper|master]` after cloning the repo, and refer to the corresponding `README.md`'s.

<img src="gym_pybullet_drones/assets/helix.gif" alt="formation flight" width="325"> <img src="gym_pybullet_drones/assets/helix.png" alt="control info" width="425">

## Installation

Tested on Intel x64/Ubuntu 22.04 and Apple Silicon/macOS 14.1.

```sh
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

```

## Use

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

### utiasDSL `pycffirmware` Python Bindings example (multiplatform, single-drone)

Install [`pycffirmware`](https://github.com/utiasDSL/pycffirmware?tab=readme-ov-file#installation) for Ubuntu, macOS, or Windows

```sh
cd gym_pybullet_drones/examples/
python3 cff-dsl.py
```

### Betaflight SITL example (Ubuntu only)

```sh
git clone https://github.com/betaflight/betaflight # use the `master` branch at the time of writing (future release 4.5)
cd betaflight/ 
make arm_sdk_install # if needed, `apt install curl``
make TARGET=SITL # comment out line: https://github.com/betaflight/betaflight/blob/master/src/main/main.c#L52
cp ~/gym-pybullet-drones/gym_pybullet_drones/assets/eeprom.bin ~/betaflight/ # assuming both gym-pybullet-drones/ and betaflight/ were cloned in ~/
betaflight/obj/main/betaflight_SITL.elf
```

In another terminal, run the example

```sh
conda activate drones
cd gym_pybullet_drones/examples/
python3 beta.py --num_drones 1 # check the steps in the file's docstrings to use multiple drones
```

## Citation

If you wish, please cite our [IROS 2021 paper](https://arxiv.org/abs/2103.02142) ([and original codebase](https://github.com/utiasDSL/gym-pybullet-drones/tree/paper)) as

```bibtex
@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      volume={},
      number={},
      pages={7512-7519},
      doi={10.1109/IROS51168.2021.9635857}
}
```

## References

- Carlos Luis and Jeroome Le Ny (2016) [*Design of a Trajectory Tracking Controller for a Nanoquadcopter*](https://arxiv.org/pdf/1608.05786.pdf)
- Nathan Michael, Daniel Mellinger, Quentin Lindsey, Vijay Kumar (2010) [*The GRASP Multiple Micro UAV Testbed*](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.169.1687&rep=rep1&type=pdf)
- Benoit Landry (2014) [*Planning and Control for Quadrotor Flight through Cluttered Environments*](http://groups.csail.mit.edu/robotics-center/public_papers/Landry15)
- Julian Forster (2015) [*System Identification of the Crazyflie 2.0 Nano Quadrocopter*](https://www.research-collection.ethz.ch/handle/20.500.11850/214143)
- Antonin Raffin, Ashley Hill, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, and Noah Dormann (2019) [*Stable Baselines3*](https://github.com/DLR-RM/stable-baselines3)
- Guanya Shi, Xichen Shi, Michael O’Connell, Rose Yu, Kamyar Azizzadenesheli, Animashree Anandkumar, Yisong Yue, and Soon-Jo Chung (2019)
[*Neural Lander: Stable Drone Landing Control Using Learned Dynamics*](https://arxiv.org/pdf/1811.08027.pdf)
- C. Karen Liu and Dan Negrut (2020) [*The Role of Physics-Based Simulators in Robotics*](https://www.annualreviews.org/doi/pdf/10.1146/annurev-control-072220-093055)
- Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza (2020) [*Flightmare: A Flexible Quadrotor Simulator*](https://arxiv.org/pdf/2009.00563.pdf)

## Core Team WIP

- [ ] Multi-drone `crazyflie-firmware` SITL support (@spencerteetaert, @JacopoPan)
- [ ] Use SITL services with steppable simulation (@JacopoPan)

## Desired Contributions/PRs

- [ ] Add motor delay, advanced ESC modeling by implementing a buffer in `BaseAviary._dynamics()`
- [ ] Replace `rpy` with quaternions (and `ang_vel` with body rates) by editing `BaseAviary._updateAndStoreKinematicInformation()`, `BaseAviary._getDroneStateVector()`, and the `.computeObs()` methods of relevant subclasses

## Troubleshooting

- On Ubuntu, with an NVIDIA card, if you receive a "Failed to create and OpenGL context" message, launch `nvidia-settings` and under "PRIME Profiles" select "NVIDIA (Performance Mode)", reboot and try again.

Run all tests from the top folder with

```sh
pytest tests/
```

-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute](https://github.com/VectorInstitute) / University of Cambridge's [Prorok Lab](https://github.com/proroklab)



##### README Gruppe Alex  Flo Moritz

BaseRLAviary_TestFlytoWall enthält den Versuch nur Vorwärts rückwärts und nichts damit wir gegen die Box wand fahren

BaseRLAviary_TestFlo enthält Ansatz mit PPO zu lernen mit den actions vor zurück links rechts

20241109_1300_MS_pid_velocity enthält die Sensorauslesung sowie auch den Versuch Mazes zu laden welche generiert und es kann über Pfeiltasten geflogen werden

Maze.py generiert die unterschiedlichen Maze URDF

learn.py verfügt über die funktion die Visualisierung nach dem Training auszugeben und den Trainingsverlauf mit rewards timestamp durchläufe... Sowie auch den Verlauf der ObervationSpace


# erster Test

V1: Reward Funktion für gegen die Wand angepasst:
>Prompt 1 Reward Flytothewall: modify the overall reward function be dependend on the distance in front State(20). Means if state20=9999 the positive-Distance reward should be 0. If the drone in flying backwards, the negative reward is -100. if the drone gets closer to the wall, the values of stat20 gets real means lower down from 10 to theretihcally 0 (means crashed into wall). The Reward should be linearly increased when the drone gets closer to the wall rising for 10 (0 reward) to 0,8 (1500 reward). Then, when the distance is between 0,8 and 0,5 the reward jumps to constant 1800 but if getting closer than 0,5 the reward instantly gets cut to -1500 (crash into wall). If the drone is in between range from 0,8 to 0,5 from the wall and not moving, the drone gets another adding reward for 200/sec. The additional will be reset to 0 if the drone moves. 


#### Befehle um es durchzuführen
python train_flytowall.py --train
python train_flytowall.py --evaluate results/save-01.09.2025_18.03.25/final_model --gui

#### ENDE Stand 09.01

File "/home/alex/Documents/RKIM/Semester_1/F&E_1/Dronnenrennen_Group/gym-pybullet-drones/gym_pybullet_drones/examples/Test_Flo/BaseAviary_TestFlytoWall.py", line 96, in __init__
    self.URDF = self.DRONE_MODEL.value + ".urdf"
AttributeError: 'str' object has no attribute 'value'


#### Session 10.01

Wir haben von dem Training und dem Stand vom 09.01 das Problem, dass die Drohne nicht nach vorne oder hinten fliegt sondern einfach nur gerade runter abstürzt

12.1.25: FT: einige Versuche gemacht, um die Drohne auf 0.5m hovern zu lassen

- Weil nichts geklappt hat, habe ich die learn.py kopiert und die BaseRLAviary_TestFlytoWall angewendet. Es lernt etwas, aber es gibt folgende Probleme:
    - wenn vorwärts an die Wand geflogen wird, wird die Env. nicht sauber zurückgesetzt
    - PPO addiert immer den ganzen Reward auf, dadurch wird der Reward viel größer als wir wollen
    - die Rewards und abstand zu wand werden geplottet: Abstand zur Wand bleibt irgendwie immer gleich...