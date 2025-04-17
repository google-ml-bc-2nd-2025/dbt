"""
다양한 3D 모델 형식 간의 뼈대(본) 매핑 정보를 제공하는 모듈
"""

# SMPL 본 이름 순서 (24개)
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
]

# Mixamo → SMPL 관절 이름 매핑
MIXAMO_TO_SMPL = {
    "Hips": "pelvis",
    "Spine": "spine1",
    "Spine1": "spine2",
    "Spine2": "spine3",
    "LeftUpLeg": "left_hip",
    "RightUpLeg": "right_hip",
    "LeftLeg": "left_knee",
    "RightLeg": "right_knee",
    "LeftFoot": "left_ankle",
    "RightFoot": "right_ankle",
    "Neck": "neck",
    "Head": "head",
    "LeftShoulder": "left_shoulder",
    "RightShoulder": "right_shoulder",
    "LeftArm": "left_elbow",
    "RightArm": "right_elbow",
    "LeftForeArm": "left_wrist",
    "RightForeArm": "right_wrist",
    "LeftHand": "left_hand",
    "RightHand": "right_hand",
    "LeftHandIndex1": "left_hand",  # 추가 매핑
    "RightHandIndex1": "right_hand"  # 추가 매핑
}

# SMPL → Mixamo 관절 이름 매핑 (역방향 매핑)
SMPL_TO_MIXAMO = {smpl_name: mixamo_name for mixamo_name, smpl_name in MIXAMO_TO_SMPL.items()}

# 여러 Mixamo 본 이름이 하나의 SMPL 본에 매핑될 수 있으므로, 우선순위 정의
MIXAMO_PRIORITY = {
    "left_hand": ["LeftHand", "LeftHandIndex1"],
    "right_hand": ["RightHand", "RightHandIndex1"]
}

def get_smpl_joint_index(joint_name):
    """SMPL 관절 이름에 해당하는 인덱스를 반환"""
    try:
        return SMPL_JOINT_NAMES.index(joint_name)
    except ValueError:
        return -1

def find_matching_bones(target_skeleton_bones, source_mapping=MIXAMO_TO_SMPL):
    """
    대상 스켈레톤의 본 이름과 소스 매핑을 비교하여 매칭되는 본 목록을 반환
    
    Args:
        target_skeleton_bones: 대상 스켈레톤의 본 이름 리스트
        source_mapping: 소스 형식에서 SMPL로의 매핑 딕셔너리
        
    Returns:
        매칭된 본 이름 쌍의 딕셔너리 {target_bone_name: source_bone_name}
    """
    matches = {}
    
    # 정확한 이름 매치 시도
    for target_bone in target_skeleton_bones:
        for source_bone, smpl_name in source_mapping.items():
            if target_bone.lower() == source_bone.lower():
                matches[target_bone] = source_bone
                break
    
    # 부분 이름 매치 시도 (정확한 매치가 없는 본에 대해)
    for target_bone in target_skeleton_bones:
        if target_bone not in matches:
            best_match = None
            best_score = 0
            
            for source_bone, smpl_name in source_mapping.items():
                # 단순 부분 문자열 매치
                if source_bone.lower() in target_bone.lower() or target_bone.lower() in source_bone.lower():
                    # 더 긴 매치를 우선시
                    score = len(source_bone)
                    if score > best_score:
                        best_score = score
                        best_match = source_bone
            
            if best_match:
                matches[target_bone] = best_match
    
    return matches