import json
import os

def save_sequence(seq, filename="data/bat_sequences.json"):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(filename):
        try:
            with open(filename) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    data.append(seq)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def load_sequences(filename="data/bat_sequences.json"):
    if os.path.exists(filename):
        with open(filename) as f:
            try:
                sequences = json.load(f)  # use json.load, not json.loads
                return sequences
            except Exception as e:
                return {"error": str(e)}
    return []

def update_stats(result, filename="data/stats.json"):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(filename):
        with open(filename) as f:
            stats = json.load(f)
    else:
        stats = {"games": 0, "wins": 0, "losses": 0, "ties": 0}

    stats["games"] += 1
    if result == "win":
        stats["wins"] += 1
    elif result == "lose":
        stats["losses"] += 1
    else:
        stats["ties"] += 1

    with open(filename, "w") as f:
        json.dump(stats, f, indent=2)
