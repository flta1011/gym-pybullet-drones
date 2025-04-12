import glob
import os

import pybullet as p
import pybullet_data
import yaml


def load_maze(maze_urdf_path, client_id):
    """Lädt ein Maze in die PyBullet-Umgebung."""
    p.resetSimulation(physicsClientId=client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)  # Boden hinzufügen
    maze_id = p.loadURDF(maze_urdf_path, physicsClientId=client_id)
    return maze_id


def spawn_drone(position, client_id):
    """Spawnt eine Drohne an der angegebenen Position."""
    drone_urdf_path = os.path.join(pybullet_data.getDataPath(), "/home/moritz_s/Documents/RKIM_1/F_u_E_Drohnenrennen/GitRepo/gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf")
    drone_id = p.loadURDF(drone_urdf_path, position, physicsClientId=client_id)
    return drone_id


def set_camera_above_drone(position, client_id):
    """Setzt die Kamera über der aktuellen Drohne."""
    camera_distance = 2  # Abstand der Kamera von der Drohne
    camera_yaw = 0  # Drehung um die vertikale Achse
    camera_pitch = -89  # Blickwinkel (Top-Down)
    camera_target = [position[0], position[1], position[2]]  # Zielpunkt der Kamera (Drohnenposition)

    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=camera_target,
        physicsClientId=client_id,
    )


def is_valid_position(position):
    """Überprüft, ob die Position gültig ist."""
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        return False
    if not all(isinstance(coord, (int, float)) for coord in position):
        return False
    return True


def main():
    # PyBullet-Client starten
    client_id = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8, physicsClientId=client_id)

    # Ordner mit URDF-Dateien angeben
    urdf_folder = "/home/moritz_s/Documents/RKIM_1/F_u_E_Drohnenrennen/GitRepo/gym-pybullet-drones/gym_pybullet_drones/assets/maze"
    yaml_path = "/home/moritz_s/Documents/RKIM_1/F_u_E_Drohnenrennen/GitRepo/gym-pybullet-drones/gym_pybullet_drones/examples/MAZE_TRAINING/Maze_init_target.yaml"

    # Alle URDF-Dateien im Ordner finden
    urdf_files = sorted(glob.glob(os.path.join(urdf_folder, "*.urdf")))

    # Sicherstellen, dass alle Dateien gefunden werden
    if not urdf_files:
        print("Keine URDF-Dateien im Ordner gefunden.")
        return

    print(f"Gefundene URDF-Dateien: {len(urdf_files)}")

    # YAML-Datei mit Startpositionen laden
    with open(yaml_path, "r") as file:
        maze_data = yaml.safe_load(file)

    # Start-Maze-Index festlegen
    start_maze_index = 21  # Ändere diesen Wert, um bei einem bestimmten Maze zu starten (z. B. 5 für das 6. Maze)

    # Mazes und Positionen durchlaufen
    for maze_index in range(start_maze_index, len(maze_data)):
        map_name = f"map{maze_index}"  # Beispiel: map0, map1, ...
        if map_name not in maze_data:
            print(f"Kein Eintrag für {map_name} in der YAML-Datei gefunden. Überspringe...")
            continue

        # Passende URDF-Datei für das aktuelle Maze finden (mit Unterstrich)
        maze_urdf_path = os.path.join(urdf_folder, f"map_{maze_index}.urdf")
        if not os.path.exists(maze_urdf_path):
            print(f"URDF-Datei für {map_name} nicht gefunden: {maze_urdf_path}. Überspringe...")
            continue

        print(f"Lade Maze {maze_index}/{len(maze_data)}: {maze_urdf_path}")

        # Maze laden
        load_maze(maze_urdf_path, client_id)

        # Positionen für das aktuelle Maze durchlaufen
        map_data = maze_data[map_name]
        if "initial_xyzs" in map_data:
            for position_index, position in enumerate(map_data["initial_xyzs"]):
                # Position validieren
                if not is_valid_position(position):
                    print(f"Ungültige Position {position} in {map_name}. Überspringe...")
                    continue

                print(f"Spawne Drohne an Position {position_index + 1}/{len(map_data['initial_xyzs'])} in {map_name}: {position}")

                # Drohne spawnen
                drone_id = spawn_drone(position, client_id)

                # Kamera über der Drohne positionieren
                set_camera_above_drone(position, client_id)

                # Warten auf Benutzereingabe
                input("Drücke Enter, um zur nächsten Position zu wechseln...")

                # Drohne entfernen
                p.removeBody(drone_id, physicsClientId=client_id)

        print(f"Maze {maze_index} abgeschlossen. Weiter mit dem nächsten Maze...")

    print("Alle Mazes und Positionen wurden überprüft.")
    p.disconnect(client_id)


if __name__ == "__main__":
    main()
