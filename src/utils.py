import logging
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

def get_logger(name: str) -> logging.Logger:
    """Sets up a standardized logger with console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler (saved in project log directory)
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "crop_yield.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def save_pickle(obj, filepath: str):
    """Saves an object to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str):
    """Loads an object from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

def save_json(data: dict, filepath: str):
    """Saves a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> dict:
    """Loads a dictionary from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

def set_plot_style():
    """Sets modern visual style for matplotlib/seaborn plots."""
    sns.set_theme(style="darkgrid")
    plt.rcParams["figure.facecolor"] = "#121212"
    plt.rcParams["axes.facecolor"] = "#1E1E1E"
    plt.rcParams["text.color"] = "#E0E0E0"
    plt.rcParams["axes.labelcolor"] = "#E0E0E0"
    plt.rcParams["xtick.color"] = "#B0B0B0"
    plt.rcParams["ytick.color"] = "#B0B0B0"
    plt.rcParams["grid.color"] = "#333333"
    plt.rcParams["axes.edgecolor"] = "#444444"
