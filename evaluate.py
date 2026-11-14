import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

TRAIN_DIR = r"D:\developement\Pickle_files\Train"
TEST_DIR = r"D:\developement\Pickle_files\Test"
GT_DIR = r"D:\developement\Pickle_files\GT"
GT_NORMAL_DIR = r"D:\developement\Pickle_files\GT_normal"

SEQ_LEN = 15

def extract_geometric_features(kpts):
    kpts = np.nan_to_num(kpts, nan=0.0, posinf=0.0, neginf=0.0)
    kpts_xy = kpts[:, :2].astype(np.float32)
    kpts_xy = np.clip(kpts_xy, -10000, 10000)
    NOSE = kpts_xy[0]; L_SHOULDER, R_SHOULDER = kpts_xy[5], kpts_xy[6]
    L_ELBOW, R_ELBOW = kpts_xy[7], kpts_xy[8]; L_WRIST, R_WRIST = kpts_xy[9], kpts_xy[10]
    L_HIP, R_HIP = kpts_xy[11], kpts_xy[12]
    shoulder_center = (L_SHOULDER + R_SHOULDER) / 2
    hip_center = (L_HIP + R_HIP) / 2
    torso = np.linalg.norm(shoulder_center - hip_center)
    if torso < 5.0 or torso > 5000.0:
        return np.zeros(10, dtype=np.float32)
    l_w_hip = np.linalg.norm(L_WRIST - L_HIP) / torso
    r_w_hip = np.linalg.norm(R_WRIST - R_HIP) / torso
    l_w_nose = np.linalg.norm(L_WRIST - NOSE) / torso
    r_w_nose = np.linalg.norm(R_WRIST - NOSE) / torso
    def angle(a, b, c):
        ba, bc = a - b, c - b
        norm_ba = np.linalg.norm(ba); norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6: return 0.0
        cos = np.dot(ba, bc) / (norm_ba * norm_bc)
        return np.arccos(np.clip(cos, -1.0, 1.0))
    l_elbow = angle(L_SHOULDER, L_ELBOW, L_WRIST)
    r_elbow = angle(R_SHOULDER, R_ELBOW, R_WRIST)
    l_rel_x = (L_WRIST[0] - hip_center[0]) / torso
    r_rel_x = (R_WRIST[0] - hip_center[0]) / torso
    features = np.array([torso, l_w_hip, r_w_hip, l_w_nose, r_w_nose, l_elbow, r_elbow, l_rel_x, r_rel_x, 0.0], dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(features, -10.0, 10.0)

class ShopliftingLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 64, 1, batch_first=True)
        self.attention = nn.Linear(64, 1)
        self.classifier = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(out), dim=1)
        ctx = torch.sum(attn * out, dim=1)
        return self.classifier(ctx).squeeze()

def load_videos(video_list):
    sequences, labels = [], []
    for filename in video_list:
        if not filename:
            continue
        train_pkl = os.path.join(TRAIN_DIR, filename)
        test_pkl = os.path.join(TEST_DIR, filename)
        if os.path.exists(train_pkl):
            pkl_path = train_pkl
            gt_path = os.path.join(GT_NORMAL_DIR, filename.replace('.pkl', '.npy'))
        elif os.path.exists(test_pkl):
            pkl_path = test_pkl
            gt_path = os.path.join(GT_DIR, filename.replace('.pkl', '.npy'))
        else:
            continue
        if not os.path.exists(gt_path):
            continue
        gt_labels = np.load(gt_path)
        with open(pkl_path, 'rb') as f:
            pose_data = pickle.load(f)
        person_buffers = {}
        for frame_idx in sorted(pose_data.keys()):
            current_label = gt_labels[frame_idx] if frame_idx < len(gt_labels) else 0
            for pid, data in pose_data[frame_idx].items():
                features = extract_geometric_features(data[1])
                if pid not in person_buffers:
                    person_buffers[pid] = []
                person_buffers[pid].append(features)
                if len(person_buffers[pid]) == SEQ_LEN:
                    sequences.append(person_buffers[pid])
                    labels.append(current_label)
                    person_buffers[pid].pop(0)
    return sequences, labels

with open('dataset_splits.txt', 'r') as f:
    content = f.read().split('\n\n')

test_videos = content[2].replace('TEST:\n', '').strip().split('\n')

print("Loading test data...")
test_seq, test_lbl = load_videos(test_videos)
print(f"Test sequences: {len(test_seq)} | Theft: {sum(test_lbl)} | Normal: {len(test_lbl) - sum(test_lbl)}")

X_test = torch.tensor(np.array(test_seq), dtype=torch.float32)
y_test = np.array(test_lbl)

model = ShopliftingLSTM()
model.load_state_dict(torch.load("poselift_final_model.pth"))
model.eval()

with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits).numpy()

preds = (probs > 0.5).astype(int)

print("\n--- FINAL EVALUATION ON TEST SET ---")
print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, preds, zero_division=0):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, probs):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))