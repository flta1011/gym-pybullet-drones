import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def process_csv_file(csv_path):
    # Datei einlesen, echte Header-Zeile ist Zeile 1 (Index 1)
    try:
        df = pd.read_csv(csv_path, skiprows=[0], header=1)
    except Exception as e:
        print(f"Fehler beim Lesen von {csv_path}: {e}")
        return False

    # Alle möglichen Spalten definieren
    all_possible_columns = [
        "Runde",
        # "Terminated",
        # "Truncated",
        "Map-Abgedeckt",
        "Wand berührungen",
        "Summe Reward",
        "Flugzeit der Runde",
        # "Maze_number",
        "Uhrzeit Welt",
    ]

    # Prüfen welche Spalten verfügbar sind
    columns_of_interest = [col for col in all_possible_columns if col in df.columns]

    # Sicherstellen dass mindestens "Runde" vorhanden ist
    if "Runde" not in columns_of_interest:
        print("Erforderliche Spalte 'Runde' nicht gefunden")
        return False

    # Nur diese Spalten behalten, falls vorhanden
    df = df[[col for col in columns_of_interest if col in df.columns]]

    if "Runde" not in df.columns:
        print(f"Spalte 'Runde' nicht in {csv_path} gefunden")
        return False

    # Runde numerisch machen und nach ihr sortieren
    df["Runde"] = pd.to_numeric(df["Runde"], errors="coerce")
    df = df.dropna(subset=["Runde"])
    df = df.sort_values(by="Runde")

    if len(df) == 0:
        print(f"Keine gültigen Daten in {csv_path}")
        return False

    # Subplots vorbereiten
    num_plots = len([col for col in columns_of_interest if col in df.columns and col != "Runde"])
    if num_plots == 0:
        print(f"Keine Daten zum Plotten in {csv_path}")
        return False

    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots), sharex=True)
    fig.suptitle(f"Auswertung über Runden - {os.path.basename(csv_path)}", fontsize=16)

    # Wenn nur ein Plot, axes in Liste umwandeln
    if num_plots == 1:
        axes = [axes]

    # Plot für jede Spalte außer "Runde"
    plot_idx = 0
        # Plot für jede Spalte außer "Runde" und die letzten zwei Spalten
    columns_to_plot = columns_of_interest[1:-2]  # skip "Runde" and last two
    for col in columns_to_plot:
        if col == "Runde" or col not in df.columns:
            continue

        ax = axes[plot_idx]
        # Prüfen, ob Spalte numerisch oder kategorisch
        try:
            y = pd.to_numeric(df[col], errors="coerce")
            if y.notna().sum() > 0:
                ax.plot(df["Runde"], y, marker="o")
            else:
                raise ValueError("Nicht-numerisch")
        except:
            ax.scatter(df["Runde"], df[col], alpha=0.7)

        ax.set_ylabel(col)
        ax.grid(True)
        plot_idx += 1

    axes[-1].set_xlabel("Runde")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # SVG-Datei speichern
    svg_path = os.path.splitext(csv_path)[0] + ".svg"
    plt.savefig(svg_path)
    plt.close(fig)
    print(f"Diagramm gespeichert: {svg_path}")
    return True


def process_folder(base_folder):
    # Alle Unterordner finden
    folders = [f for f in glob.glob(os.path.join(base_folder, "*")) if os.path.isdir(f)]
    folders.append(base_folder)  # Auch das Basisverzeichnis verarbeiten

    total_processed = 0

    for folder in folders:
        print(f"Verarbeite Ordner: {folder}")
        # Alle CSV-Dateien im Ordner finden
        csv_files = glob.glob(os.path.join(folder, "*.csv"))

        for csv_file in csv_files:
            # Prüfen, ob bereits eine SVG-Datei existiert
            svg_file = os.path.splitext(csv_file)[0] + ".svg"
            if not os.path.exists(svg_file):
                print(f"Verarbeite CSV: {csv_file}")
                if process_csv_file(csv_file):
                    total_processed += 1
            else:
                print(f"SVG existiert bereits für: {csv_file}")

    print(f"Insgesamt {total_processed} CSV-Dateien verarbeitet und SVG-Dateien erstellt.")


# Hauptordner angeben und Verarbeitung starten
base_folder = os.path.dirname(os.path.abspath(__file__))
process_folder(base_folder)
