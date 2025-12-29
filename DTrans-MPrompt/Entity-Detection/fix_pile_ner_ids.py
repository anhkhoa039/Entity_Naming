import json
import os

def fix_ids(filename):
    path = f"dataset/{filename}"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Fixing IDs in {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for idx, entry in enumerate(data):
        entry['id'] = idx  # Replace with integer ID

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Fixed {len(data)} entries in {path}")

if __name__ == "__main__":
    fix_ids("pile_ner_train.json")
    fix_ids("pile_ner_dev.json")
    fix_ids("pile_ner_test.json")
