import json

def load_training_data(file_path: str):
    """
    Loads a JSONL file and converts it to a list of (job_title, raw_text, label) tuples.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append((obj["job_title"], obj["raw_text"], int(obj["label"])))
    return data