import csv
import os
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

import matplotlib.pyplot as plt
import numpy as np


class MazeGenerator:
    """
    Erstellt aus einer Maze *.CSV Datei eine URDF Datei
    """

    def __init__(self, width=3, height=3, discretization=0.05):
        self.width = int(width / discretization)  # Maze width in grid units
        self.height = int(height / discretization)  # Maze height in grid units
        self.discretization = discretization  # Grid resolution
        self.grid = np.zeros((self.height, self.width), dtype=int)  # Maze grid initialized as empty

    def generate_from_csv(self, filename):
        """Generate a maze from a CSV file."""
        self.grid = np.genfromtxt(filename, delimiter=",", dtype=int)

    def _add_wall_link(self, root, name, size, xyz, rpy="0 0 0", mass=1):
        """Add a link to the URDF file."""
        link = ET.SubElement(root, "link", name=name)
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        box = ET.SubElement(geometry, "box", size=size)
        origin = ET.SubElement(visual, "origin", xyz=xyz, rpy=rpy)
        inertial = ET.SubElement(link, "inertial")
        mass_element = ET.SubElement(inertial, "mass", value=str(mass))
        inertia = ET.SubElement(inertial, "inertia", ixx="1", iyy="1", izz="1")
        collision = ET.SubElement(link, "collision")
        collision_geometry = ET.SubElement(collision, "geometry")
        collision_box = ET.SubElement(collision_geometry, "box", size=size)
        collision_origin = ET.SubElement(collision, "origin", xyz=xyz, rpy=rpy)

    def _add_joint(self, root, name, parent, child, xyz="0 0 0", rpy="0 0 0"):
        """Add a joint to the URDF file."""
        joint = ET.SubElement(root, "joint", name=name, type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)

    def generate_urdf_from_maze(self, maze_height=1, urdf_filename="maze.urdf"):

        # Add the root element
        root = ET.Element("robot", name="maze")

        # Add the base link
        base_link = ET.SubElement(root, "link", name="base_link")
        # Add inertial to base_link
        inertial = ET.SubElement(base_link, "inertial")
        ET.SubElement(inertial, "mass", value="1")
        ET.SubElement(inertial, "inertia", ixx="1", iyy="1", izz="1")

        # Add the floor
        floor_size = f"{self.width * self.discretization} {self.height * self.discretization} {self.discretization}"
        floor_xyz = f"{self.width * self.discretization / 2} {self.height * self.discretization / 2} -{self.discretization / 2}"
        self._add_wall_link(root, "floor", floor_size, floor_xyz, mass=1)
        self._add_joint(root, "floor_joint", "base_link", "floor")

        # # Add the base link
        # base_link = ET.SubElement(root, "link", name="base_link")

        # # Add the floor
        # floor_size = f"{self.width * self.discretization} {self.height * self.discretization} {self.discretization}"
        # floor_xyz = f"{self.width * self.discretization / 2} {self.height * self.discretization / 2} -{self.discretization / 2}"
        # self._add_wall_link(root, "floor", floor_size, floor_xyz)
        # self._add_joint(root, "floor_joint", "base_link", "floor")

        # Add the walls as a single link
        wall_link = ET.SubElement(root, "link", name="walls")

        # Initialize total mass and inertia
        total_mass = 0
        total_ixx = 0
        total_iyy = 0
        total_izz = 0

        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == 1:
                    wall_size = f"{self.discretization} {self.discretization} {maze_height}"
                    wall_xyz = f"{x * self.discretization + self.discretization / 2} {y * self.discretization + self.discretization / 2} {maze_height / 2}"

                    # Add visual geometry
                    visual = ET.SubElement(wall_link, "visual")
                    geometry = ET.SubElement(visual, "geometry")
                    box = ET.SubElement(geometry, "box", size=wall_size)
                    origin = ET.SubElement(visual, "origin", xyz=wall_xyz, rpy="0 0 0")

                    # Add collision geometry
                    collision = ET.SubElement(wall_link, "collision")
                    collision_geometry = ET.SubElement(collision, "geometry")
                    collision_box = ET.SubElement(collision_geometry, "box", size=wall_size)
                    collision_origin = ET.SubElement(collision, "origin", xyz=wall_xyz, rpy="0 0 0")

                    # Parse size (width, height, depth)
                    width, height, depth = map(float, wall_size.split())

                    # Calculate mass and inertia for this wall
                    mass = 1  # Set mass for each wall
                    ixx = (1 / 12) * mass * (height**2 + depth**2)
                    iyy = (1 / 12) * mass * (width**2 + depth**2)
                    izz = (1 / 12) * mass * (width**2 + height**2)

                    # Accumulate total mass and inertia
                    total_mass += mass
                    total_ixx += ixx
                    total_iyy += iyy
                    total_izz += izz

        # Add inertial element to the walls link
        inertial = ET.SubElement(wall_link, "inertial")
        ET.SubElement(inertial, "mass", value=str(total_mass))
        ET.SubElement(inertial, "inertia", ixx=str(total_ixx), iyy=str(total_iyy), izz=str(total_izz))

        # Add a joint for the walls
        self._add_joint(root, "walls_joint", "base_link", "walls", xyz="0 0 0")

        # Write the URDF file
        tree = ET.ElementTree(root)
        xml_str = ET.tostring(root, encoding="unicode")
        pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
        with open(urdf_filename, "w") as f:
            f.write(pretty_xml_str)


if __name__ == "__main__":
    maze_height = 1  # Height of the maze walls
    csv_directory = "gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps"  # Replace with the path to your CSV files
    urdf_directory = "gym_pybullet_drones/assets/maze/20250320"  # Directory to save URDF files

    if not os.path.exists(urdf_directory):
        os.makedirs(urdf_directory)

    for csv_file in os.listdir(csv_directory):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(csv_directory, csv_file)
            urdf_filename = os.path.splitext(csv_file)[0] + ".urdf"
            urdf_path = os.path.join(urdf_directory, urdf_filename)

            maze = MazeGenerator()  # Optional: set a seed for reproducibility
            maze.generate_from_csv(csv_path)
            maze.generate_urdf_from_maze(maze_height=1, urdf_filename=urdf_path)
