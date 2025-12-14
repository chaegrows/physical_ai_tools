import json
import os

file_path = '/media/metafarmers/7440017c-39b5-476a-a24b-4fb0a9c24140/chae/dualArm_rosbag/physical_ai_tools/lerobot/outputs/train/checkpoint-17000/pretrained_model/config.json'

with open(file_path, 'r') as f:
    config = json.load(f)

config['embodiment_tag'] = 'new_embodiment'

with open(file_path, 'w') as f:
    json.dump(config, f, indent=2)

print("Updated config.json with embodiment_tag: new_embodiment")
