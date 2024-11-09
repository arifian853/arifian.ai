import json

# Load the existing dataset
input_path = '/mnt/data/data.json'
output_path = '/mnt/data/multi_turn_dialog.json'

with open(input_path, 'r') as file:
    data = json.load(file)

# Convert dataset into multi-turn dialogs
multi_turn_dialogs = []

# Grouping related questions into a dialog session
dialog_buffer = []
previous_person = "Arifian"
for entry in data:
    user_input = entry['prompt']
    bot_response = entry['response']

    # Append conversation turns to buffer
    dialog_buffer.append({"user": user_input, "bot": bot_response})

    # Check for new dialog initiation or end of data to create a session
    if len(dialog_buffer) >= 3 or entry == data[-1]:  # Example limit for multi-turn
        multi_turn_dialogs.append({"dialog": dialog_buffer})
        dialog_buffer = []

# Save augmented multi-turn dialog dataset
with open(output_path, 'w') as file:
    json.dump(multi_turn_dialogs, file, indent=4, ensure_ascii=False)

output_path
