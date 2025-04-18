smpl_bone_list = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist"
    # "left_hand", # 학습 데이터는 22개. 손목까지만 존재함.
    # "right_hand"
]

# 관절 연결을 올바른 계층 구조로 수정
joint_connections = [
    # 척추 계층 - 이름이 같은 관절끼리 연결
    (0, 3),    # pelvis -> spine1 (인덱스로 직접 연결)
    (3, 6),    # spine1 -> spine2
    (6, 11),   # spine2 -> spine3
    (11, 14),  # spine3 -> neck
    (14, 17),  # neck -> head
    
    # 왼쪽 팔 계층 - 왼쪽 어깨는 척추3이 아닌 어깨에 연결
    (6, 9),    # spine2 -> left_shoulder (spine3 대신 spine2에 연결)
    (9, 12),   # left_shoulder -> left_elbow
    (12, 15),  # left_elbow -> left_wrist
    (15, 18),  # left_wrist -> left_hand
    
    # 오른쪽 팔 계층
    (6, 10),   # spine2 -> right_shoulder (spine3 대신 spine2에 연결)
    (10, 13),  # right_shoulder -> right_elbow
    (13, 16),  # right_elbow -> right_wrist
    (16, 19),  # right_wrist -> right_hand
    
    # 왼쪽 다리 계층
    (0, 1),    # pelvis -> left_hip
    (1, 4),    # left_hip -> left_knee
    (4, 7),    # left_knee -> left_ankle
    (7, 20),   # left_ankle -> left_foot
    (20, 22),  # left_foot -> left_toe
    
    # 오른쪽 다리 계층
    (0, 2),    # pelvis -> right_hip
    (2, 5),    # right_hip -> right_knee
    (5, 8),    # right_knee -> right_ankle
    (8, 21),   # right_ankle -> right_foot
    (21, 23)   # right_foot -> right_toe
]


# SMPL 관절 이름 (순서대로)
smpl_joint_names = [
    "pelvis", 
    "left_hip", 
    "right_hip", 
    "spine1", 
    "left_knee", 
    "right_knee", 
    "spine2", 
    "left_ankle", 
    "right_ankle", 
    "left_shoulder", 
    "right_shoulder", 
    "spine3", 
    "left_elbow", 
    "right_elbow", 
    "neck", 
    "left_wrist", 
    "right_wrist", 
    "head", 
    "left_hand", 
    "right_hand", 
    "left_foot", 
    "right_foot", 
    "left_toe", 
    "right_toe"
]

# MIXAMO to SMPL 관절 매핑 ( MIXAMO -> SMPL )
mixamo_to_smpl_map = {
    "mixamorig:Hips": "pelvis",
    "mixamorig:LeftUpLeg": "left_hip",
    "mixamorig:RightUpLeg": "right_hip",
    "mixamorig:Spine": "spine1",
    "mixamorig:LeftLeg": "left_knee",
    "mixamorig:RightLeg": "right_knee",
    "mixamorig:Spine1": "spine2",
    "mixamorig:LeftFoot": "left_ankle",
    "mixamorig:RightFoot": "right_ankle",
    "mixamorig:Spine2": "spine3",
    "mixamorig:LeftToeBase": "left_foot",
    "mixamorig:RightToeBase": "right_foot",
    "mixamorig:Neck": "neck",
    "mixamorig:LeftShoulder": "left_collar",
    "mixamorig:RightShoulder": "right_collar",
    "mixamorig:Head": "head",
    "mixamorig:LeftArm": "left_shoulder",
    "mixamorig:RightArm": "right_shoulder",
    "mixamorig:LeftForeArm": "left_elbow",
    "mixamorig:RightForeArm": "right_elbow",
    "mixamorig:LeftHand": "left_wrist",
    "mixamorig:RightHand": "right_wrist",
    # "mixamorig:LeftHandMiddle1": "left_hand",
    # "mixamorig:RightHandMiddle1": "right_hand"
}

# SMPL to MIXAMO 관절 매핑 (SMPL -> MIXAMO)
smpl_to_mixamo_map = {
    "pelvis": "mixamorig:Hips",
    "left_hip": "mixamorig:LeftUpLeg",
    "right_hip": "mixamorig:RightUpLeg",
    "spine1": "mixamorig:Spine",
    "left_knee": "mixamorig:LeftLeg",
    "right_knee": "mixamorig:RightLeg",
    "spine2": "mixamorig:Spine1",
    "left_ankle": "mixamorig:LeftFoot",
    "right_ankle": "mixamorig:RightFoot",
    "spine3": "mixamorig:Spine2",
    "left_foot": "mixamorig:LeftToeBase",
    "right_foot": "mixamorig:RightToeBase",
    "neck": "mixamorig:Neck",
    "left_collar": "mixamorig:LeftShoulder",
    "right_collar": "mixamorig:RightShoulder",
    "head": "mixamorig:Head",
    "left_shoulder": "mixamorig:LeftArm",
    "right_shoulder": "mixamorig:RightArm",
    "left_elbow": "mixamorig:LeftForeArm",
    "right_elbow": "mixamorig:RightForeArm",
    # "left_wrist": "mixamorig:LeftHand",
    # "right_wrist": "mixamorig:RightHand",
}


# 다양한 접두사를 가진 모델 대응을 위한 확장 매핑 생성 함수
def create_extended_mappings(base_map, prefixes=None):
    """
    기본 매핑을 바탕으로 다양한 접두사를 가진 확장 매핑 생성
    """
    if prefixes is None:
        prefixes = ["", "mixamorig:", "mixamorig_", "mixamo:", "mixamo_", "m_"]
    
    extended_map = {}
    
    for smpl_name, mixamo_name in base_map.items():
        # mixamorig: 접두사 제거하여 기본 이름 추출
        base_name = mixamo_name
        for prefix in ["mixamorig:", "mixamorig_", "mixamo:", "mixamo_"]:
            if mixamo_name.startswith(prefix):
                base_name = mixamo_name[len(prefix):]
                break
        
        # 각 접두사로 새 매핑 생성
        for prefix in prefixes:
            extended_map[smpl_name] = f"{prefix}{base_name}"
    
    return extended_map

# 확장 매핑 생성 및 원본 매핑과 병합
extended_smpl_to_mixamo = create_extended_mappings(smpl_to_mixamo_map)
smpl_to_mixamo_map.update(extended_smpl_to_mixamo)

# glb는 인덱스화 되어 있지 않음. 라벨로 매칭할 것!