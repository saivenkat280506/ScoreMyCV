#This code is generated with AI to create my own dataset for Learning and Testing purpose

import json
from sklearn.model_selection import train_test_split

# Load original full dataset
with open("train_data.json", "r") as f:
    data = json.load(f)

# 80% train, 20% test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save new train_data.json
with open("train_data.json", "w") as f:
    json.dump(train_data, f, indent=2)

# Save test_data.json
with open("test_data.json", "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Split complete: {len(train_data)} train | {len(test_data)} test")
