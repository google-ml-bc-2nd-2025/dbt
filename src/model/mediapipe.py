'''

Mediapipe Index	Mediapipe Name	SMPL Joint	SMPL Index	비고
0	nose	head	15	대략적 위치 매칭
11	left_shoulder	left_collar	13	
12	right_shoulder	right_collar	14	
13	left_elbow	left_elbow	18	
14	right_elbow	right_elbow	19	
15	left_wrist	left_wrist	20	
16	right_wrist	right_wrist	21	
23	left_hip	left_hip	1	
24	right_hip	right_hip	2	
25	left_knee	left_knee	4	
26	right_knee	right_knee	5	
27	left_ankle	left_ankle	7	
28	right_ankle	right_ankle	8	
31	left_foot_index	left_foot	10	대략적 대응
32	right_foot_index	right_foot	11	대략적 대응
1~10	얼굴 관련	없음	-	생략 가능
17~22	손가락	없음	-	SMPL은 손가락 없음
29~30	heel (뒤꿈치)	없음	-	SMPL은 발가락 포함 없음

'''

mediapipe_bone_list = [
    "nose", 
    "left_eye_inner", 
    "left_eye", 
    "left_eye_outer", 
    "right_eye_inner", 
    "right_eye", 
    "right_eye_outer",
    "left_ear", 
    "right_ear", 
    "mouth_left", 
    "mouth_right", 
    "left_shoulder", 
    "right_shoulder", 
    "left_elbow",
    "right_elbow", 
    "left_wrist", 
    "right_wrist", 
    "left_pinky", 
    "right_pinky", 
    "left_index", 
    "right_index",
    "left_thumb", 
    "right_thumb", 
    "left_hip", 
    "right_hip", 
    "left_knee", 
    "right_knee", 
    "left_ankle",
    "right_ankle", 
    "left_heel", 
    "right_heel", 
    "left_foot_index", 
    "right_foot_index"
]

# MediaPipe 랜드마크 인덱스 정보 (참고용)
mp_landmarks_info = {
    0: "nose", 
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

mediapipe_to_smpl_map = {
    0: 15,     # nose → head
    11: 13,    # left_shoulder → left_collar
    12: 14,    # right_shoulder → right_collar
    13: 18,    # left_elbow → left_elbow
    14: 19,    # right_elbow → right_elbow
    15: 20,    # left_wrist → left_wrist
    16: 21,    # right_wrist → right_wrist
    23: 1,     # left_hip → left_hip
    24: 2,     # right_hip → right_hip
    25: 4,     # left_knee → left_knee
    26: 5,     # right_knee → right_knee
    27: 7,     # left_ankle → left_ankle
    28: 8,     # right_ankle → right_ankle
    31: 10,    # left_foot_index → left_foot
    32: 11     # right_foot_index → right_foot
}