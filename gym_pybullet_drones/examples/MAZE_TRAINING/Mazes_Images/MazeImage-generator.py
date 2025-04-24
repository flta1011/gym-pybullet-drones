import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ordner, in dem das Skript liegt
script_dir = os.path.dirname(os.path.abspath(__file__))

# ----------------------
# 1. PNG-Dateien löschen
# ----------------------
for file in os.listdir(script_dir):
    if file.endswith(".png"):
        try:
            os.remove(os.path.join(script_dir, file))
        except Exception as e:
            print(f"Fehler beim Löschen von {file}: {e}")

# ----------------------
# 2. CSVs in Bilder umwandeln
# ----------------------
csv_files = [f for f in os.listdir(script_dir) if f.endswith(".csv")]

for csv_file in csv_files:
    file_path = os.path.join(script_dir, csv_file)

    # Leere Felder als 0 interpretieren
    df = pd.read_csv(file_path, header=None, dtype=str).fillna("0")
    df = df.replace(r"^\s*$", "0", regex=True)
    array = df.astype(int).to_numpy()

    # 1 → schwarz (0), 0 → weiß (255)
    image_array = np.where(array == 1, 0, 255).astype(np.uint8)

    # Bild vertikal spiegeln
    flipped_image = cv2.flip(image_array, 0)

    # Bild speichern
    image_name = f"{os.path.splitext(csv_file)[0]}.png"
    cv2.imwrite(os.path.join(script_dir, image_name), flipped_image)

# ----------------------
# 3. Übersichtsgrafik erstellen
# ----------------------
image_files = sorted([f for f in os.listdir(script_dir) if f.endswith(".png") and "map_" in f], key=lambda x: int("".join(filter(str.isdigit, x))))

fig, axes = plt.subplots(6, 5, figsize=(20, 24))
# Increase vertical spacing between subplots
plt.subplots_adjust(hspace=0.4)
axes = axes.flatten()

for ax, image_file in zip(axes, image_files):
    img_path = os.path.join(script_dir, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    maze_number = "".join(filter(str.isdigit, image_file))
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Labyrinth {maze_number}", fontsize=18)
    ax.axis("off")

# Nicht genutzte Achsen verbergen
for i in range(len(image_files), len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "maze_übersicht.png"))
