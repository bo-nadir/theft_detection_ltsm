import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

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
        return torch.sigmoid(self.classifier(ctx)).squeeze()

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

VIDEO_PATH = 'test_video.mp4'
OUTPUT_PATH = 'final_output.mp4'
FRAME_SKIP = 3
SEQ_LEN = 15

print("Loading models...")
pose_model = YOLO('yolov8n-pose.pt')
lstm_model = ShopliftingLSTM()
lstm_model.load_state_dict(torch.load("poselift_final_model.pth"))
lstm_model.eval()

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

person_sequences = {}
frame_count = 0

print("Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    
    if frame_count % FRAME_SKIP != 0:
        out.write(frame)
        continue
    
    results = pose_model.track(frame, persist=True, verbose=False)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        keypoints = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for i, pid in enumerate(track_ids):
            p_kpts = keypoints[i]
            if p_kpts[9][0] == 0 and p_kpts[10][0] == 0:
                continue
            
            features = extract_geometric_features(p_kpts)
            
            if pid not in person_sequences:
                person_sequences[pid] = []
            person_sequences[pid].append(features)
            
            if len(person_sequences[pid]) > SEQ_LEN:
                person_sequences[pid].pop(0)
            
            if len(person_sequences[pid]) == SEQ_LEN:
                seq_tensor = torch.tensor(np.array([person_sequences[pid]]), dtype=torch.float32)
                with torch.no_grad():
                    theft_prob = lstm_model(seq_tensor).item()
                
                x1, y1, x2, y2 = map(int, boxes[i])
                
                if theft_prob > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    text = f"THEFT: {theft_prob:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw, y1), (0, 0, 0), -1)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"Normal: {theft_prob:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 0), -1)
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    out.write(frame)

cap.release()
out.release()
print(f"✅ Done! Video saved as {OUTPUT_PATH}")