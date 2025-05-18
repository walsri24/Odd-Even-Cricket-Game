import json
import os

def save_sequence(seq, filename="data/bat_sequences.json"):
    os.makedirs("data", exist_ok=True)
    with open(filename, "a") as f:
        json.dump(seq, f)
        f.write("\n")

def load_sequences(filename="data/bat_sequences.json"):
    sequences = []
    if os.path.exists(filename):
        with open(filename) as f:
            for line in f:
                try:
                    sequences.append(json.loads(line))
                except:
                    continue
    return sequences

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
