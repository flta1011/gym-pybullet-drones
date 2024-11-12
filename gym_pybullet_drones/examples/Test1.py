import pybullet as p
import pybullet_data
import time

# Initialisierung und Plane-Umgebung laden
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane_id = p.loadURDF("plane.urdf")

# Drohne laden
initial_height = 1.0
drone_id = p.loadURDF("/Users/floriantausch/Library/Mobile Documents/com~apple~CloudDocs/Referenzmaterial/Master-Studium/HKA Hochschule Karlsruhe Masterstudium Robotik und KI in der Produktion RKIM/F&E- Projekt 1 - Drohnenrennen/Github-Repo/gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf", [0, 0, initial_height])

# Simulationsparameter
time_step = 1./240.
p.setTimeStep(time_step)

# Initiale Kameraeinstellungen
camera_distance = 1
camera_yaw = 50
camera_pitch = -35
camera_target_position = [0, 0, 1]

p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)


# PID-Parameter
Kp = 1.0
Ki = 0.0
Kd = 0.0
integral = 0.0
previous_error = 0.0

desired_height = initial_height



while True:
    # Aktuelle Höhe abrufen
    pos, orn = p.getBasePositionAndOrientation(drone_id)
    current_height = pos[2]

    # Fehler berechnen
    error = desired_height - current_height
    integral += error * time_step
    derivative = (error - previous_error) / time_step
    previous_error = error

    # PID-Regler anwenden
    thrust = Kp * error + Ki * integral + Kd * derivative

    # Tastatureingaben erfassen
    keys = p.getKeyboardEvents()

    # Steuerbefehle initialisieren
    force_x = 0.0
    force_y = 0.0

    # Bewegungssteuerung basierend auf Tastatureingaben
    if ord('w') in keys:
        force_y = 5.0  # Vorwärts
    if ord('s') in keys:
        force_y = -5.0  # Rückwärts
    if ord('a') in keys:
        force_x = -5.0  # Links
    if ord('d') in keys:
        force_x = 5.0  # Rechts

    # Holen Sie die Orientierung der Drohne
    _, drone_orientation = p.getBasePositionAndOrientation(drone_id)
    drone_rotation_matrix = p.getMatrixFromQuaternion(drone_orientation)

    # Transformieren Sie die Kräfte in das lokale Koordinatensystem der Drohne
    local_force = [
        force_x * drone_rotation_matrix[0] + force_y * drone_rotation_matrix[1],
        force_x * drone_rotation_matrix[3] + force_y * drone_rotation_matrix[4],
        force_x * drone_rotation_matrix[6] + force_y * drone_rotation_matrix[7]
    ]

    # Kräfte anwenden
    p.applyExternalForce(drone_id, -1, local_force, [0, 0, 0], p.LINK_FRAME)

    # Simulation vorantreiben
    p.stepSimulation()
    time.sleep(time_step)
    
    # Innerhalb der Simulationsschleife
    drone_pos, _ = p.getBasePositionAndOrientation(drone_id)
    camera_target_position = drone_pos

    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)