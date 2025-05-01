"""
다양한 3D 모델 형식 간의 뼈대(본) 매핑 정보를 제공하는 모듈
"""
from model.smpl import smpl_bone_list as SMPL_JOINT_NAMES

# Mixamo → SMPL 관절 이름 매핑 - 더 다양한 이름 패턴 추가
MIXAMO_TO_SMPL = {
    # 기본 매핑
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
    "LeftHandIndex1": "left_hand",
    "RightHandIndex1": "right_hand",
    
    # mixamorig: 접두사 추가된 경우 (대부분의 Mixamo 모델)
    "mixamorig:Hips": "pelvis",
    "mixamorig:Spine": "spine1",
    "mixamorig:Spine1": "spine2",
    "mixamorig:Spine2": "spine3",
    "mixamorig:LeftUpLeg": "left_hip",
    "mixamorig:RightUpLeg": "right_hip",
    "mixamorig:LeftLeg": "left_knee",
    "mixamorig:RightLeg": "right_knee",
    "mixamorig:LeftFoot": "left_ankle",
    "mixamorig:RightFoot": "right_ankle",
    "mixamorig:Neck": "neck",
    "mixamorig:Head": "head",
    "mixamorig:LeftShoulder": "left_shoulder",
    "mixamorig:RightShoulder": "right_shoulder",
    "mixamorig:LeftArm": "left_elbow",
    "mixamorig:RightArm": "right_elbow",
    "mixamorig:LeftForeArm": "left_wrist",
    "mixamorig:RightForeArm": "right_wrist",
    "mixamorig:LeftHand": "left_hand",
    "mixamorig:RightHand": "right_hand",
    "mixamorig:LeftHandIndex1": "left_hand",
    "mixamorig:RightHandIndex1": "right_hand",
    
    # 다른 명명 규칙
    "hip": "pelvis",
    "spine": "spine1",
    "spine_1": "spine2",
    "spine_2": "spine3",
    "left_leg_upper": "left_hip",
    "right_leg_upper": "right_hip",
    "left_leg_lower": "left_knee",
    "right_leg_lower": "right_knee",
    "left_foot_joint": "left_ankle",
    "right_foot_joint": "right_ankle",
    "neck_joint": "neck",
    "head_joint": "head",
    "left_arm_upper": "left_elbow",
    "right_arm_upper": "right_elbow",
    "left_arm_lower": "left_wrist",
    "right_arm_lower": "right_wrist",
}

# SMPL → Mixamo 관절 이름 매핑 (역방향 매핑)
SMPL_TO_MIXAMO = {smpl_name: mixamo_name for mixamo_name, smpl_name in MIXAMO_TO_SMPL.items() if "mixamorig:" in mixamo_name}

# 여러 Mixamo 본 이름이 하나의 SMPL 본에 매핑될 수 있으므로, 우선순위 정의
MIXAMO_PRIORITY = {
    "left_hand": ["LeftHand", "LeftHandIndex1", "mixamorig:LeftHand", "mixamorig:LeftHandIndex1"],
    "right_hand": ["RightHand", "RightHandIndex1", "mixamorig:RightHand", "mixamorig:RightHandIndex1"]
}

def find_matching_bones(target_skeleton_bones, source_mapping=MIXAMO_TO_SMPL, nodes=None):
    """
    대상 스켈레톤의 본 이름과 소스 매핑을 비교하여 매칭되는 본 목록을 반환
    
    Args:
        target_skeleton_bones: 대상 스켈레톤의 본 이름 리스트
        source_mapping: 소스 형식에서 SMPL로의 매핑 딕셔너리
        nodes: GLB 노드 리스트 (계층 구조 탐색에 사용)
        
    Returns:
        매칭된 본 이름 쌍의 딕셔너리 {target_bone_name: source_bone_name}
    """
    matches = {}
    
    # 매핑 정보 디버깅 출력
    print(f"타겟 본 개수: {len(target_skeleton_bones)}, 소스 매핑 개수: {len(source_mapping)}")
    
    # 1단계: 정확한 이름 매치 (현재와 동일)
    for target_bone in target_skeleton_bones:
        target_bone_lower = target_bone.lower()
        for source_bone, smpl_name in source_mapping.items():
            if target_bone_lower == source_bone.lower():
                matches[target_bone] = smpl_name
                print(f"정확한 매칭: {target_bone} -> {smpl_name}")
                break
    
    # 2단계: 더 다양한 부분 매칭 시도
    for target_bone in target_skeleton_bones:
        if target_bone in matches:
            continue
            
        best_match = None
        best_score = 0
        target_bone_lower = target_bone.lower()
        
        # 접두사와 숫자 제거 변환 (더 유연한 매칭을 위함)
        target_simple = target_bone_lower
        prefixes = ["mixamorig:", "mixamorig_", "mixamo:", "mixamo_", "m_", "joint_"]
        for prefix in prefixes:
            if target_simple.startswith(prefix):
                target_simple = target_simple[len(prefix):]
        
        # 숫자와 언더스코어 제거
        target_simple = ''.join([c for c in target_simple if not c.isdigit() and c != '_'])
        
        # 공통 단어 매칭 (left, right, arm, leg, shoulder 등)
        common_keywords = {
            "left": "left", "right": "right", "arm": "arm", "leg": "leg", "hand": "hand", 
            "foot": "foot", "shoulder": "shoulder", "hip": "hip", "spine": "spine", 
            "head": "head", "neck": "neck", "ankle": "ankle", "knee": "knee",
            "wrist": "wrist", "elbow": "elbow", "collar": "collar"
        }
        
        # 좌우 방향 처리
        is_left = "left" in target_simple or "l_" in target_simple or target_simple.startswith("l")
        is_right = "right" in target_simple or "r_" in target_simple or target_simple.startswith("r")
        
        for source_bone, smpl_name in source_mapping.items():
            source_bone_lower = source_bone.lower()
            source_simple = source_bone_lower
            
            # 접두사 제거
            for prefix in prefixes:
                if source_simple.startswith(prefix):
                    source_simple = source_simple[len(prefix):]
            
            # 숫자와 언더스코어 제거
            source_simple = ''.join([c for c in source_simple if not c.isdigit() and c != '_'])
            
            # 1. 접두사 제거 후 완전 일치
            if target_simple == source_simple:
                best_match = smpl_name
                best_score = 1000  # 높은 점수
                break
            
            # 2. 부분 문자열 매치 (글자 길이 기준)
            if source_simple in target_simple or target_simple in source_simple:
                score = len(source_simple) if source_simple in target_simple else len(target_simple)
                if score > best_score:
                    best_score = score
                    best_match = smpl_name
            
            # 3. 좌우 매칭 + 부위 매칭 (예: leftarm -> left_elbow)
            if is_left and "left" in smpl_name:
                for keyword in common_keywords:
                    if keyword in target_simple and keyword in smpl_name:
                        score = 200 + len(keyword)  # 방향성 + 부위 매치는 높은 점수
                        if score > best_score:
                            best_score = score
                            best_match = smpl_name
            
            if is_right and "right" in smpl_name:
                for keyword in common_keywords:
                    if keyword in target_simple and keyword in smpl_name:
                        score = 200 + len(keyword)  # 방향성 + 부위 매치는 높은 점수
                        if score > best_score:
                            best_score = score
                            best_match = smpl_name
        
        if best_match:
            matches[target_bone] = best_match
            print(f"부분 매칭: {target_bone} -> {best_match} (점수: {best_score})")
        else:
            print(f"매칭 실패: {target_bone} -> 없음")
    
    # 3단계: 계층 구조 활용 매칭
    if nodes:
        print("계층 구조 기반 매칭 시작...")
        
        # 부모-자식 관계 구축
        child_to_parent = {}
        for i, node in enumerate(nodes):
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    child_to_parent[child] = i
        
        # 노드 인덱스 -> 본 이름 매핑
        node_to_bone = {}
        for i, node in enumerate(nodes):
            if hasattr(node, 'name') and node.name in target_skeleton_bones:
                node_to_bone[i] = node.name
        
        # 부모/자식 본 매핑 전파
        for node_idx, bone_name in node_to_bone.items():
            if bone_name in matches:
                continue  # 이미 매핑된 본은 건너뜀
            
            # 부모 노드 매핑 확인
            parent_idx = child_to_parent.get(node_idx)
            depth = 0
            max_depth = 3  # 최대 3단계까지 거슬러 올라감
            
            # 부모 방향 탐색
            while parent_idx is not None and depth < max_depth:
                if parent_idx in node_to_bone:
                    parent_bone = node_to_bone[parent_idx]
                    if parent_bone in matches:
                        # 부모의 매핑을 기반으로 현재 본 매핑 추측
                        parent_smpl = matches[parent_bone]
                        # 부모 SMPL 본에 기반하여 자식 본 매핑 유추
                        if "left" in parent_smpl and "left" in bone_name.lower():
                            for smpl_name in SMPL_JOINT_NAMES:
                                if "left" in smpl_name and not any(b for b in matches.values() if b == smpl_name):
                                    matches[bone_name] = smpl_name
                                    print(f"계층 기반 매핑: {bone_name} -> {smpl_name} (부모: {parent_bone})")
                                    break
                        elif "right" in parent_smpl and "right" in bone_name.lower():
                            for smpl_name in SMPL_JOINT_NAMES:
                                if "right" in smpl_name and not any(b for b in matches.values() if b == smpl_name):
                                    matches[bone_name] = smpl_name
                                    print(f"계층 기반 매핑: {bone_name} -> {smpl_name} (부모: {parent_bone})")
                                    break
                        break
                parent_idx = child_to_parent.get(parent_idx)
                depth += 1
    
    # 최종 매핑 결과
    print(f"총 {len(matches)}개의 본 매핑 완료 (타겟 본의 {len(matches)/len(target_skeleton_bones)*100:.1f}%)")
    return matches