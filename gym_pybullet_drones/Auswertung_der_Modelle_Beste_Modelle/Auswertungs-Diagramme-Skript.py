import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def create_plots(df, columns_to_plot, base_path, title_suffix="", maze_number=None, csv_name=""):
    """Helper function to create plots for a given dataframe
    Args:
        df: DataFrame with the data
        columns_to_plot: List of columns to plot
        base_path: Base path for saving files
        title_suffix: Additional text for the title
        maze_number: If set, indicates this is a maze-specific plot
        csv_name: Name of the CSV file
    """
    num_plots = len(columns_to_plot)
    if num_plots == 0:
        return False

    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots), sharex=True)

    # Create main title
    title = f"Auswertung über Runden{title_suffix}"
    if maze_number is not None:
        title += f" - Maze {maze_number}"

    # Add title and subtitle with different font sizes
    fig.text(0.5, 0.98, title, fontsize=16, ha="center", va="bottom")
    fig.text(0.5, 0.95, csv_name, fontsize=12, ha="center", va="top", style="italic")

    # Wenn nur ein Plot, axes in Liste umwandeln
    if num_plots == 1:
        axes = [axes]

    # Plot für jede Spalte außer "Runde"
    for idx, col in enumerate(columns_to_plot):
        ax = axes[idx]
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

    axes[-1].set_xlabel("Runde")
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # Bestimme Dateinamen-Suffix basierend auf maze_number
    file_suffix = f"_maze_{maze_number}" if maze_number is not None else ""

    # SVG-Datei speichern
    svg_path = base_path + file_suffix + ".svg"
    plt.savefig(svg_path, bbox_inches="tight", format="svg")

    # PNG-Datei speichern
    png_path = base_path + file_suffix + ".png"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)

    plt.close(fig)
    print(f"Diagramme gespeichert: {svg_path} und {png_path}")
    return True


def process_csv_file(csv_path):
    # Datei einlesen
    try:
        # First try to read the file to find the actual header row
        with open(csv_path, "r") as file:
            lines = file.readlines()

        # Find the line that contains our main column headers
        header_row = -1
        for i, line in enumerate(lines):
            if "Runde" in line and ("Map-Abgedeckt" in line or "Flugzeit" in line):
                header_row = i
                break

        if header_row == -1:
            print(f"Keine Kopfzeile mit 'Runde' und 'Map-Abgedeckt' oder 'Flugzeit' gefunden in {csv_path}")
            return False

        print(f"Gefundene Kopfzeile in Zeile {header_row + 1}: {lines[header_row].strip()}")

        # Now read the CSV with the correct header row
        df = pd.read_csv(csv_path, skiprows=range(header_row), header=0)
        print(f"Gefundene Spalten: {', '.join(df.columns)}")

    except Exception as e:
        print(f"Fehler beim Lesen von {csv_path}: {e}")
        return False

    # Alle möglichen Spalten definieren
    all_possible_columns = ["Runde", "Map-Abgedeckt", "Wand berührungen", "Summe Reward", "Flugzeit der Runde", "Maze_number"]  # Added Maze_number to possible columns

    # Prüfen welche Spalten verfügbar sind
    columns_of_interest = [col for col in all_possible_columns if col in df.columns]
    print(f"Verfügbare Spalten: {', '.join(columns_of_interest)}")

    # Sicherstellen dass mindestens die erforderlichen Spalten vorhanden sind
    required_columns = ["Runde", "Map-Abgedeckt"]
    missing_columns = [col for col in required_columns if col not in columns_of_interest]
    if missing_columns:
        print(f"Fehlende erforderliche Spalten: {', '.join(missing_columns)}")
        return False

    # Nur diese Spalten behalten, falls vorhanden
    df = df[[col for col in columns_of_interest if col in df.columns]]

    # Spalten numerisch machen
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except:
            print(f"Warnung: Konnte Spalte {col} nicht in numerische Werte umwandeln")

    # Filter für Map-Abgedeckt > 0 und gültige Runden
    original_len = len(df)
    df = df[(df["Map-Abgedeckt"] > 0.0) & (df["Runde"].notna())]
    print(f"Filterung: {original_len} Zeilen -> {len(df)} Zeilen")

    # Zusätzlicher Filter für Flugzeit, falls vorhanden
    if "Flugzeit der Runde" in df.columns:
        original_len = len(df)
        df = df[df["Flugzeit der Runde"] > 0]
        print(f"Flugzeit-Filterung: {original_len} Zeilen -> {len(df)} Zeilen")

    df = df.sort_values(by="Runde")

    if len(df) == 0:
        print(f"Keine gültigen Daten in {csv_path} (nach Filterung)")
        return False

    # Base path for saving files
    base_path = os.path.splitext(csv_path)[0]
    csv_name = os.path.basename(csv_path)

    # Columns to plot (excluding Runde and Maze_number)
    columns_to_plot = [col for col in columns_of_interest if col not in ["Runde", "Maze_number"]]

    # Create overall plot
    create_plots(df, columns_to_plot, base_path, title_suffix=" - Gesamt", csv_name=csv_name)

    # If Maze_number column exists, create separate plots for each maze
    if "Maze_number" in df.columns:
        unique_mazes = df["Maze_number"].unique()
        print(f"Gefundene Maze-Nummern: {', '.join(map(str, unique_mazes))}")

        for maze in unique_mazes:
            maze_df = df[df["Maze_number"] == maze]
            if len(maze_df) > 0:
                create_plots(maze_df, columns_to_plot, base_path, title_suffix="", maze_number=maze, csv_name=csv_name)

    return True


def process_folder(base_folder):
    # Alle Unterordner finden
    folders = [f for f in glob.glob(os.path.join(base_folder, "*")) if os.path.isdir(f)]
    folders.append(base_folder)  # Auch das Basisverzeichnis verarbeiten

    total_processed = 0

    for folder in folders:
        print(f"Verarbeite Ordner: {folder}")

        # Lösche alle existierenden SVG- und PNG-Dateien
        existing_files = glob.glob(os.path.join(folder, "*.svg")) + glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))
        for file in existing_files:
            try:
                os.remove(file)
                print(f"Gelöschte Datei: {file}")
            except Exception as e:
                print(f"Fehler beim Löschen von {file}: {e}")

        # Alle CSV-Dateien im Ordner finden
        csv_files = glob.glob(os.path.join(folder, "*.csv"))

        for csv_file in csv_files:
            print(f"Verarbeite CSV: {csv_file}")
            if process_csv_file(csv_file):
                total_processed += 1

    print(f"Insgesamt {total_processed} CSV-Dateien verarbeitet und Diagramme erstellt.")


# Hauptordner angeben und Verarbeitung starten
base_folder = os.path.dirname(os.path.abspath(__file__))
process_folder(base_folder)
