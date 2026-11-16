import os
import numpy as np
import random

TRAIN_DIR = r"D:\developement\Pickle_files\Train"
TEST_DIR = r"D:\developement\Pickle_files\Test"
GT_DIR = r"D:\developement\Pickle_files\GT"

random.seed(42)

# Separate test videos into theft and normal
theft_videos = []
normal_from_test = []

for filename in os.listdir(TEST_DIR):
    if not filename.endswith('.pkl'):
        continue
    gt_name = filename.replace('.pkl', '.npy')
    gt_path = os.path.join(GT_DIR, gt_name)
    if os.path.exists(gt_path):
        labels = np.load(gt_path)
        if np.sum(labels) > 0:
            theft_videos.append(filename)
        else:
            normal_from_test.append(filename)

# All train videos + normal test videos
normal_videos = normal_from_test + [f for f in os.listdir(TRAIN_DIR) if f.endswith('.pkl')]

print(f"Theft videos (from test): {len(theft_videos)}")
print(f"Normal videos (train + normal test): {len(normal_videos)}")

def split_list(lst, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(lst)
    n = len(lst)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return lst[:train_end], lst[train_end:val_end], lst[val_end:]

train_theft, val_theft, test_theft = split_list(theft_videos)
train_normal, val_normal, test_normal = split_list(normal_videos)

final_train = train_theft + train_normal
final_val = val_theft + val_normal
final_test = test_theft + test_normal

with open('dataset_splits.txt', 'w') as f:
    f.write("TRAIN:\n" + "\n".join(final_train) + "\n\n")
    f.write("VAL:\n" + "\n".join(final_val) + "\n\n")
    f.write("TEST:\n" + "\n".join(final_test) + "\n")

print(f"\n--- SPLIT RESULTS ---")
print(f"Train: {len(final_train)} ({len(train_theft)} theft, {len(train_normal)} normal)")
print(f"Val:   {len(final_val)} ({len(val_theft)} theft, {len(val_normal)} normal)")
print(f"Test:  {len(final_test)} ({len(test_theft)} theft, {len(test_normal)} normal)")
print(f"\nSaved to dataset_splits.txt")