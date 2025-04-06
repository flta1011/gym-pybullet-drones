import pandas as pd
import matplotlib.pyplot as plt

# Datei einlesen, echte Header-Zeile ist Zeile 1 (Index 1)
df = pd.read_csv("gym_pybullet_drones/Auswertung_der_Modelle_Archieve/M5_R4_O7_A2_20250402-103344.csv", skiprows=[0], header=1)

# Benötigte Spalten
columns_of_interest = [
    "Runde", 
    #"Terminated", 
    #"Truncated", 
    "Map-Abgedeckt", 
    "Wand berührungen", 
    "Summe Reward",
    #"Maze_number",
    #"Uhrzeit"
]

# Nur diese Spalten behalten, falls vorhanden
df = df[[col for col in columns_of_interest if col in df.columns]]

# Runde numerisch machen und nach ihr sortieren
df["Runde"] = pd.to_numeric(df["Runde"], errors='coerce')
df = df.dropna(subset=["Runde"])
df = df.sort_values(by="Runde")

# Subplots vorbereiten
num_plots = len(columns_of_interest) - 1  # ohne "Runde"
fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots), sharex=True)
fig.suptitle("Auswertung über Runden", fontsize=16)

# Plot für jede Spalte außer "Runde"
plot_idx = 0
for col in columns_of_interest:
    if col == "Runde" or col not in df.columns:
        continue

    ax = axes[plot_idx]
    # Prüfen, ob Spalte numerisch oder kategorisch
    try:
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().sum() > 0:
            ax.plot(df["Runde"], y, marker='o')
        else:
            raise ValueError("Nicht-numerisch")
    except:
        ax.scatter(df["Runde"], df[col], alpha=0.7)

    ax.set_ylabel(col)
    ax.grid(True)
    plot_idx += 1

axes[-1].set_xlabel("Runde")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()