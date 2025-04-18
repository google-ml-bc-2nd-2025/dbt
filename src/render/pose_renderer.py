"""
포즈 데이터 시각화 및 렌더링 모듈
"""

import numpy as np
import cv2
import json
import plotly.graph_objects as go
import plotly.io as pio

# 포즈 미리보기 관련 함수들은 남겨두되, 사용하지 않도록 수정
# 나중에 필요할 경우 다시 활성화할 수 있도록 코드는 유지

def get_skeleton_connections(num_joints):
    """
    관절 수에 따라 적절한 스켈레톤 연결 구조를 반환합니다.
    
    Args:
        num_joints: 관절의 수
        
    Returns:
        연결 구조 리스트 [(인덱스1, 인덱스2, 이름), ...]
    """
    if num_joints == 33:  # MediaPipe Pose (33개 랜드마크)
        return [
            # 얼굴
            (0, 1, "Face/0-1"),
            (1, 2, "Face/1-2"),
            (2, 3, "Face/2-3"),
            (3, 7, "Face/3-7"),
            (0, 4, "Face/0-4"),
            (4, 5, "Face/4-5"),
            (5, 6, "Face/5-6"),
            (6, 8, "Face/6-8"),
            
            # 몸통
            (9, 10, "Spine/Shoulder"),
            (11, 12, "Spine/Hip"),
            (11, 23, "Spine/Left-Hip"),
            (12, 24, "Spine/Right-Hip"),
            (9, 11, "Spine/Left-Body"),
            (10, 12, "Spine/Right-Body"),
            
            # 왼쪽 팔
            (11, 13, "LeftArm/Shoulder-Elbow"),
            (13, 15, "LeftArm/Elbow-Wrist"),
            (15, 17, "LeftHand/Wrist-Pinky"),
            (15, 19, "LeftHand/Wrist-Index"),
            (15, 21, "LeftHand/Wrist-Thumb"),
            (17, 19, "LeftHand/Pinky-Index"),
            
            # 오른쪽 팔
            (12, 14, "RightArm/Shoulder-Elbow"),
            (14, 16, "RightArm/Elbow-Wrist"),
            (16, 18, "RightHand/Wrist-Pinky"),
            (16, 20, "RightHand/Wrist-Index"),
            (16, 22, "RightHand/Wrist-Thumb"),
            (18, 20, "RightHand/Pinky-Index"),
            
            # 왼쪽 다리
            (23, 25, "LeftLeg/Hip-Knee"),
            (25, 27, "LeftLeg/Knee-Ankle"),
            (27, 29, "LeftFoot/Ankle-Heel"),
            (29, 31, "LeftFoot/Heel-Toe"),
            (27, 31, "LeftFoot/Ankle-Toe"),
            
            # 오른쪽 다리
            (24, 26, "RightLeg/Hip-Knee"),
            (26, 28, "RightLeg/Knee-Ankle"),
            (28, 30, "RightFoot/Ankle-Heel"),
            (30, 32, "RightFoot/Heel-Toe"),
            (28, 32, "RightFoot/Ankle-Toe")
        ]
    elif num_joints == 17:  # COCO 포맷 (17개 관절)
        return [
            (0, 1, "Spine/Neck-RShoulder"),
            (0, 2, "Spine/Neck-LShoulder"),
            (1, 3, "RightArm/Shoulder-Elbow"),
            (3, 5, "RightArm/Elbow-Wrist"),
            (2, 4, "LeftArm/Shoulder-Elbow"),
            (4, 6, "LeftArm/Elbow-Wrist"),
            (0, 7, "Spine/Neck-Hip"),
            (7, 8, "Spine/Hip-RHip"),
            (7, 9, "Spine/Hip-LHip"),
            (8, 10, "RightLeg/Hip-Knee"),
            (10, 12, "RightLeg/Knee-Ankle"),
            (9, 11, "LeftLeg/Hip-Knee"),
            (11, 13, "LeftLeg/Knee-Ankle"),
            (0, 14, "Head/Neck-Nose"),
            (14, 16, "Head/Nose-REye"),
            (14, 15, "Head/Nose-LEye")
        ]
    else:  # 기본 골격 구조 (몸만)
        return [
            (0, 1, "Spine/Center-Neck"),
            (1, 2, "Head/Neck-Head"),
            (0, 3, "Spine/Center-LShoulder"),
            (3, 4, "LeftArm/Shoulder-Elbow"),
            (4, 5, "LeftArm/Elbow-Wrist"),
            (0, 6, "Spine/Center-RShoulder"),
            (6, 7, "RightArm/Shoulder-Elbow"),
            (7, 8, "RightArm/Elbow-Wrist"),
            (0, 9, "Spine/Center-LHip"),
            (9, 10, "LeftLeg/Hip-Knee"),
            (10, 11, "LeftLeg/Knee-Ankle"),
            (0, 12, "Spine/Center-RHip"),
            (12, 13, "RightLeg/Hip-Knee"),
            (13, 14, "RightLeg/Knee-Ankle")
        ]