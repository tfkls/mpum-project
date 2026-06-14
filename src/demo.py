"""
Disclaimer: this was in large part created with help of AI for presentation purposes.

Live demo usage:
  python demo.py enroll <name>   capture repetitions
  python demo.py predict         capture one input and predict
"""

import pickle
import sys
import tkinter as tk

import numpy as np
import pandas as pd

from main import ROOT, load_data
from models.triplet import MLPEmbedding

PASSWORD = ".tie5Roanl"
KEYS = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]
KEYSYMS = {
    "period": "period",
    "t": "t",
    "i": "i",
    "e": "e",
    "5": "five",
    "R": "Shift.r",
    "o": "o",
    "a": "a",
    "n": "n",
    "l": "l",
    "Return": "Return",
}
REPS_FILE = ROOT / "results" / "demo_reps.csv"
MODEL_FILE = ROOT / "cache" / "demo_model.pkl"


def features(pressed):
    row = {}
    for i, (key, down, up) in enumerate(pressed):
        row[f"H.{key}"] = up - down
        if i > 0:
            row[f"UD.{pressed[i - 1][0]}.{key}"] = down - pressed[i - 1][2]
    return row


def embed(model, df):
    x = df[load_data().columns.drop("subject")].to_numpy()
    return model._embed((x - model.mus) / model.sigmas)


def get_model():
    if MODEL_FILE.exists():
        return pickle.loads(MODEL_FILE.read_bytes())
    print("Training the embedding on the CMU subjects (one-time)...")
    model = MLPEmbedding().train(load_data())
    MODEL_FILE.write_bytes(pickle.dumps(model))
    return model


class Capture:
    def __init__(self, title, action):
        self.action = action
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("640x240")
        tk.Label(self.root, text=PASSWORD, font=("monospace", 60)).pack(pady=10)
        self.typed = tk.Label(self.root, text="", font=("monospace", 60))
        self.typed.pack()
        self.status = tk.Label(
            self.root, text="Type the password, end with Enter", font=("monospace", 20)
        )
        self.status.pack(pady=10)
        self.root.bind("<KeyPress>", self.press)
        self.root.bind("<KeyRelease>", self.release)
        self.reset()
        self.root.mainloop()

    def reset(self):
        self.pressed = []
        self.open = {}
        self.typed.config(text="")

    def press(self, event):
        if "Shift" in event.keysym:
            return
        key = KEYSYMS.get(event.keysym)
        if len(self.pressed) >= len(KEYS) or key != KEYS[len(self.pressed)]:
            self.status.config(text="Please start over")
            self.reset()
            return
        record = [key, event.time / 1000, None]
        self.pressed.append(record)
        self.open[event.keycode] = record
        self.typed.config(text=PASSWORD[: len(self.pressed)])

    def release(self, event):
        record = self.open.pop(event.keycode, None)
        if record:
            record[2] = event.time / 1000
        if len(self.pressed) == len(KEYS) and not self.open:
            message = self.action(features(self.pressed))
            self.status.config(text=message)
            self.reset()


def enroll(name):
    def save(row):
        pd.DataFrame([{"name": name, **row}]).to_csv(
            REPS_FILE, mode="a", header=not REPS_FILE.exists(), index=False
        )
        count = (pd.read_csv(REPS_FILE)["name"] == name).sum()
        return f"saved - {count} repetitions for {name}"

    Capture(f"Enroll {name}", save)


def predict():
    model = get_model()
    reps = pd.read_csv(REPS_FILE)
    names = sorted(reps["name"].unique())
    centroids = np.array(
        [embed(model, reps[reps["name"] == n]).mean(axis=0) for n in names]
    )
    print(f"Enrolled: {', '.join(names)}")

    def identify(row):
        e = embed(model, pd.DataFrame([row]))
        distances = ((centroids - e) ** 2).sum(axis=1)
        order = distances.argsort()
        verdict = (
            names[order[0]]
            + f"(runner-up {names[order[1]]}, margin {distances[order[1]] / distances[order[0]]:.2f}x)"
            if len(names) > 1
            else ""
        )
        print(verdict, " dists:", {n: round(d, 2) for n, d in zip(names, distances)})
        return verdict

    Capture("Predict", identify)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "enroll":
        enroll(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "predict":
        predict()
    else:
        print(__doc__)
