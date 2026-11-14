import os
import pickle
import numpy as np

TRAIN_DIR = r"D:\developement\Pickle_files\Train"
GT_NORMAL_DIR = r"D:\developement\Pickle_files\GT_normal"

os.makedirs(GT_NORMAL_DIR, exist_ok=True)

count = 0
for filename in os.listdir(TRAIN_DIR):
    if not filename.endswith('.pkl'):
        continue
    
    pkl_path = os.path.join(TRAIN_DIR, filename)
    with open(pkl_path, 'rb') as f:
        pose_data = pickle.load(f)
    
    if len(pose_data) == 0:
        continue
    
    num_frames = max(pose_data.keys()) + 1
    labels = np.zeros(num_frames, dtype=int)
    
    save_name = filename.replace('.pkl', '.npy')
    save_path = os.path.join(GT_NORMAL_DIR, save_name)
    np.save(save_path, labels)
    count += 1

print(f"✅ Created {count} label files in {GT_NORMAL_DIR}")