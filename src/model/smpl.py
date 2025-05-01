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
    # "left_hand", # 학습 데이터(humanml3d)는 22개. 손목까지만 존재함.
    # "right_hand"
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

smpl_humanml3d_to_mixamo_index = [
    "mixamorig:Hips",           # 0 + 
    "mixamorig:LeftUpLeg",      # 1 + 
    "mixamorig:RightUpLeg",     # 2
    "mixamorig:Spine",          # 3 
    "mixamorig:LeftLeg",        # 4
    "mixamorig:RightLeg",       # 5
    "mixamorig:Spine1",         # 6 
    "mixamorig:LeftFoot",       # 7
    "mixamorig:RightFoot",      # 8
    "mixamorig:Spine2",         # 9
    "mixamorig:LeftToeBase",    # 10
    "mixamorig:RightToeBase",   # 11
    "mixamorig:Neck",           # 12 +
    "mixamorig:LeftShoulder",   # 13
    "mixamorig:RightShoulder",  # 14
    "mixamorig:Head",           # 15
    "mixamorig:LeftArm",        # 16
    "mixamorig:RightArm",       # 17
    "mixamorig:LeftForeArm",    # 18
    "mixamorig:RightForeArm",   # 19
    "mixamorig:LeftHand",       # 20
    "mixamorig:RightHand",      # 21
]
