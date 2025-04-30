"""
학습 데이터 수정 탭을 위한 기능 모듈입니다.
"""

import gradio as gr
import os
import numpy as np
from pathlib import Path
import glob
import json
import re
from datetime import datetime
import torch
from util import text_parser
from model.smpl import smpl_humanml3d_to_mixamo_index

# 파일 경로를 파일 객체처럼 처리하기 위한 클래스 추가
class FilePathWrapper:
    def __init__(self, path):
        self.name = path
        self.path = path

def create_dataset_create_tab(models_dir):
    """
    학습 데이터 생성 탭 UI를 생성합니다.
    
    Args:
        models_dir: glb,fbx 파일이 저장된 디렉토리 경로
    """
    # 데이터셋 디렉토리 경로 설정
    dataset_dir = Path(models_dir).parent.parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    # fbx/glb 파일을 찾는 함수
    def find_org_files(directory):
        # FBX와 GLB 파일 확장자에 대한 패턴 생성
        fbx_files = glob.glob(str(Path(directory) / "**" / "*.fbx"), recursive=True)
        glb_files = glob.glob(str(Path(directory) / "**" / "*.glb"), recursive=True)
        
        # 찾은 모든 파일을 합치고 정렬해서 반환
        return sorted(fbx_files + glb_files)

    # MotionClip NPZ 파일을 찾는 함수
    def find_npz_files(directory):
        return sorted(glob.glob(str(Path(directory) / "*_motionclip.npz")))
    
    def get_npz_info(file_path):
        try:
            data = np.load(file_path, allow_pickle=True)
            text = data.get('text')
            print(f"NPZ 파일에서 텍스트 데이터 로드: {text}")
            text_str = text_parser.decode_tagged(text[0]) if isinstance(text, np.ndarray) and text.size > 0 else "설명 없음"
            print(f"NPZ 파일에서 텍스트 데이터: {text_str}")
            # 포즈 데이터 형태 확인
            pose_shape = str(data.get('pose').shape) if 'pose' in data else "데이터 없음"
            
            # 파일명에서 애니메이션 이름 추출
            filename = os.path.basename(file_path)
            animation_name = filename.replace("_motionclip.npz", "")
            
            return {
                "path": file_path,
                "name": animation_name,
                "text": text_str,
                "shape": pose_shape
            }
        except Exception as e:
            print(e)
            return {
                "path": file_path,
                "name": os.path.basename(file_path),
                "text": f"오류: {str(e)}",
                "shape": "오류"
            }

    # GLB/FBX 파일 정보 추출 함수 수정
    def get_3d_file_info(file_path):
        try:
            # 파일명에서 애니메이션 이름 추출
            filename = os.path.basename(file_path)
            animation_name = Path(filename).stem
            
            # 파일 확장자 확인
            file_ext = Path(file_path).suffix.lower()
            
            # 파일 사이즈 확인
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB 단위
            
            # 동일한 이름의 텍스트 파일이 있는지 확인
            txt_file_path = str(Path(file_path).with_suffix('.txt'))
            description_text = ""
            
            # 텍스트 파일이 있으면 항상 새로 읽기
            if os.path.exists(txt_file_path):
                try:
                    # 현재 시간 기반으로 캐시 무효화를 위해 파일 수정 시간 확인
                    last_modified = os.path.getmtime(txt_file_path)
                    print(f"텍스트 파일 마지막 수정 시간: {last_modified}")
                    
                    # 파일을 매번 새로 읽음
                    with open(txt_file_path, 'r', encoding='utf-8') as f:
                        description_text = f.read().strip()
                    print(f"텍스트 파일에서 설명을 로드했습니다: {txt_file_path}, 내용: {description_text}")
                except Exception as e:
                    print(f"텍스트 파일 읽기 오류: {e}")
            
            # 텍스트 파일이 없거나 비어있으면 파일명에서 설명 생성
            if not description_text:
                # 파일명에서 확장자 제거하고 특수문자로 분할
                name_parts = re.split(r'[ _\-.]', animation_name)
                # 빈 요소 제거하고 쉼표로 조인
                description_text = ', '.join([part for part in name_parts if part])
                print(f"파일명에서 설명을 생성했습니다: {description_text}")
            
            # 파일 유형에 따른 처리
            if file_ext == '.glb':
                try:
                    # GLB 파일 처리
                    import pygltflib
                    gltf = pygltflib.GLTF2().load(file_path)
                    
                    # 애니메이션 정보
                    if gltf.animations and len(gltf.animations) > 0:
                        anim = gltf.animations[0]
                        anim_name = getattr(anim, 'name', 'Unnamed')
                        channel_count = len(anim.channels) if hasattr(anim, 'channels') else 0
                        
                        # 노드 수 확인
                        node_count = len(gltf.nodes) if hasattr(gltf, 'nodes') else 0
                        
                        shape_info = f"노드: {node_count}, 채널: {channel_count}"
                        
                        # 텍스트 설명이 없으면 애니메이션 이름 사용
                        if not description_text and anim_name != 'Unnamed':
                            description_text = f"Animation: {anim_name}"
                    else:
                        shape_info = f"크기: {file_size:.2f} MB"
                        if not description_text:
                            description_text = "GLB 모델 (애니메이션 정보 없음)"
                except Exception as e:
                    # GLB 파싱 실패 시 기본 정보만 표시
                    shape_info = f"크기: {file_size:.2f} MB"
                    if not description_text:
                        description_text = "GLB 모델 (메타데이터 로드 실패)"
                    print(f"GLB 파일 메타데이터 로드 오류: {e}")
            elif file_ext == '.fbx':
                # FBX 파일 처리 - 외부 라이브러리 없이 기본 정보만 반환
                try:
                    # FBX 파일은 바이너리 형식이라 메타데이터를 쉽게 추출하기 어려움
                    # 필요하다면 FBX SDK나 Blender Python API 등의 전문 도구 사용 필요
                    
                    # 파일 헤더만 읽어서 FBX 여부 확인 (첫 20바이트만)
                    with open(file_path, 'rb') as f:
                        header = f.read(20)
                        is_valid_fbx = header.startswith(b'Kaydara FBX Binary')
                    
                    shape_info = f"크기: {file_size:.2f} MB"
                    if not description_text:
                        if is_valid_fbx:
                            description_text = "FBX 모델"
                        else:
                            description_text = "유효하지 않은 FBX 파일일 수 있음"
                except Exception as e:
                    shape_info = f"크기: {file_size:.2f} MB"
                    if not description_text:
                        description_text = "FBX 모델 (읽기 실패)"
                    print(f"FBX 파일 읽기 오류: {e}")
            else:
                # 기타 파일 형식
                shape_info = f"크기: {file_size:.2f} MB"
                if not description_text:
                    description_text = f"지원되지 않는 파일 형식: {file_ext}"
                
            print(f"파일 정보: {file_path}, 이름: {animation_name}, 설명: {description_text}, 형태: {shape_info}")
            print(f'정리된 텍스트 {text_parser.decode_tagged(description_text)}')
            return {
                "path": file_path,
                "name": animation_name,
                "text": text_parser.decode_tagged(description_text),
                "shape": shape_info
            }
        except Exception as e:
            print(e)
            return {
                "path": file_path,
                "name": os.path.basename(file_path),
                "text": f"오류: {str(e)}",
                "shape": "오류"
            }

    # 데이터셋 목록 업데이트
    def update_dataset_list(directory):
        files = find_org_files(directory)
        print(f"찾은 파일: {files}")
        file_infos = []
        
        # 각 파일 처리
        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext in ['.glb', '.fbx']:
                # GLB/FBX 파일 처리
                file_info = get_3d_file_info(file)
            else:
                # NPZ 파일 처리
                file_info = get_npz_info(file)
            
            file_infos.append(file_info)
        
        # Gradio 테이블에 맞는 형식으로 변환
        rows = [[info["name"], info["text"], info["shape"], info["path"]] for info in file_infos]
        
        return rows
    
    # 설명 텍스트 업데이트 함수 수정
    def update_description(file_path, new_description):
        print(f"파일 경로: {file_path}, 설명: {new_description}")
        try:
            if not file_path or not os.path.exists(file_path):
                return "파일을 선택해주세요."
                
            if not new_description.strip():
                return "설명을 입력해주세요."
                
            # 파일 확장자 확인
            file_ext = Path(file_path).suffix.lower()
            
            # NPZ 파일인 경우
            if file_ext == '.npz':
                try:
                    # 기존 npz 파일에 설명 업데이트
                    data = np.load(file_path, allow_pickle=True)
                    file_data = {k: data[k] for k in data.keys()}
                    
                    # 설명 텍스트 업데이트
                    print(f"NPZ 업데이트트: {new_description}")
                    file_data['text'] = np.array([new_description.strip()], dtype=object)
                    
                    # 임시 파일로 저장
                    temp_path = file_path + ".temp"
                    np.savez(temp_path, **file_data)
                    
                    # 원본 파일 교체
                    os.replace(temp_path, file_path)
                    return f"NPZ 파일의 설명이 업데이트 되었습니다: '{new_description}'"
                except Exception as e:
                    print(e)
                    return f"NPZ 파일 업데이트 중 오류 발생: {str(e)}"
            
            # GLB/FBX 파일인 경우 - txt 파일에 설명 저장
            elif file_ext in ['.glb', '.fbx']:
                # 동일한 이름의 txt 파일 경로 생성
                txt_file_path = str(Path(file_path).with_suffix('.txt'))
                
                try:
                    # 텍스트 파일에 설명 저장
                    with open(txt_file_path, 'w', encoding='utf-8') as f:
                        print(f'new_description = {new_description}')
                        full_description = text_parser.encode_tagged(new_description.strip())
                        f.write(full_description)
                    
                    return f"{Path(file_path).stem} 모델의 설명이 저장되었습니다. ({txt_file_path})"
                except Exception as e:
                    print(e)
                    return f"텍스트 파일 저장 중 오류 발생: {str(e)}"
            
            # 지원되지 않는 파일 형식
            else:
                return f"지원되지 않는 파일 형식: {file_ext}"
                    
        except Exception as e:
            print(e)
            return f"오류 발생: {str(e)}"
    
    # 설명 업데이트 후 새로고침을 하나의 함수로 결합
    def update_and_refresh(file_path, new_description, directory, continue_repeat=False):
        """
        설명을 업데이트하고 즉시 목록을 새로고침하는 통합 함수
        
        Args:
            file_path: 설명을 업데이트할 파일 경로
            new_description: 새로운 설명 텍스트
            directory: 데이터셋 디렉토리 경로
            continue_repeat: 작업 후 계속 반복할지 여부
        """
        # 먼저 설명 업데이트
        update_result = update_description(file_path, new_description)
        
        # 그 다음 데이터셋 목록 새로고침
        updated_rows = update_dataset_list(directory)
        
        # 계속 반복 체크박스가 선택된 경우
        if continue_repeat:
            next_action = "✓ 계속 반복이 선택되었습니다. 다음 파일을 선택하여 작업을 계속하세요."
            return f"{update_result}\n\n{next_action}", updated_rows
        else:
            return update_result, updated_rows

    # 선택한 파일의 학습 데이터(NPZ) 생성 함수 추가
    def generate_training_file(file_path, description):
        try:
            if not file_path or not os.path.exists(file_path):
                return "파일을 선택해주세요."
            
            # 파일 확장자 확인
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in ['.glb', '.fbx']:
                return f"지원되지 않는 파일 형식: {file_ext} (GLB 또는 FBX 파일만 지원)"
            
            # 출력 파일 경로 생성
            output_name = f"{Path(file_path).stem}_motionclip.npz"
            output_dir = Path(file_path).parent
            output_path = output_dir / output_name
            
            # 설명 텍스트가 비어있으면 파일 이름에서 생성
            if not description.strip():
                name_parts = re.split(r'[ _\-.]', Path(file_path).stem)
                description = ', '.join([part for part in name_parts if part])
            
            try:
                # 여기서 실제 GLB/FBX 변환 코드 구현 필요
                # 임시 구현: 더미 데이터 생성
                frame_count = 60  # MotionCLIP 기본 프레임 수
                
                # MotionCLIP 포맷에 맞게 데이터 생성
                pose = np.zeros((frame_count, 132), dtype=np.float32)  # [60, 132] 형태의 6D 포즈   [frame, 22, 3]
                trans = np.zeros((frame_count, 3), dtype=np.float32)   # [60, 3] 형태의 위치
                betas = np.zeros(10, dtype=np.float32)                 # [10] 형태의 체형 파라미터
                
                # NPZ 파일 저장
                np.savez(
                    output_path,
                    pose=pose,
                    trans=trans,
                    betas=betas,
                    text=np.array([text_parser.encode_tagged(description.strip())], dtype=object)
                )
                
                return f"MotionCLIP 학습 파일이 생성되었습니다: {output_path}"
            except Exception as e:
                print(e)
                return f"학습 파일 생성 중 오류 발생: {str(e)}"
        except Exception as e:
            print(e)
            return f"오류 발생: {str(e)}"

    # NPZ 파일 생성 후 새로고침을 하나의 함수로 결합
    def generate_and_refresh(file_path, new_description, directory, continue_repeat=False):
        """
        학습 파일을 생성하고 즉시 목록을 새로고침하는 통합 함수
        
        Args:
            file_path: 학습 파일을 생성할 원본 파일 경로
            new_description: 학습 파일에 저장할 설명 텍스트
            directory: 데이터셋 디렉토리 경로
            continue_repeat: 작업 후 계속 반복할지 여부
        """
        # 먼저 NPZ 파일 생성
        generate_result = generate_training_file(file_path, new_description)
        
        # 그 다음 데이터셋 목록 새로고침
        updated_rows = update_dataset_list(directory)
        
        # 계속 반복 체크박스가 선택된 경우
        if continue_repeat:
            next_action = "✓ 계속 반복이 선택되었습니다. 다음 파일을 선택하여 작업을 계속하세요."
            return f"{generate_result}\n\n{next_action}", updated_rows
        else:
            return generate_result, updated_rows

    def get_glb_info(file_path):
        import pygltflib
        from struct import unpack
        from model.smpl import smpl_humanml3d_to_mixamo_index  # SMPL 본 이름 리스트

        try:
            gltf = pygltflib.GLTF2().load(file_path)
            if gltf.animations and len(gltf.animations) > 0:

                # 첫 번째 애니메이션 정보
                anim = gltf.animations[0]
                time_values = []
                frame_count = 0
                frame_rate = 30.0  # 기본 프레임 레이트 (필요시 수정 가능)
                # 타임스탬프 데이터 추출하여 프레임 레이트 계산
                if anim.samplers and len(anim.samplers) > 0:
                    # 첫 번째 샘플러에서 시간 데이터 추출
                    time_sampler = anim.samplers[0]
                    time_accessor = gltf.accessors[time_sampler.input]
                    time_data = get_accessor_data(gltf, time_accessor)
                    
                    if time_data is not None and len(time_data) > 1:
                        # 첫 번째와 마지막 타임스탬프
                        start_time = time_data[0]
                        end_time = time_data[-1]
                        duration = end_time - start_time
                        
                        # 타임스탬프의 개수로 프레임 수 결정
                        frame_count = len(time_data)
                        
                        # 전체 지속 시간에 따른 프레임 레이트 계산 (프레임 간격이 일정하다고 가정)
                        if duration > 0:
                            frame_rate = (frame_count - 1) / duration
                            print(f"애니메이션 지속 시간: {duration:.2f}초, 프레임 수: {frame_count}, 프레임 레이트: {frame_rate:.2f}fps")
                        else:
                            # 지속 시간이 0이면 기본값 사용
                            print(f"애니메이션 지속 시간이 0입니다. 기본 프레임 레이트 {frame_rate}fps 사용")
                    else:
                        # 샘플러에 시간 데이터가 없으면 출력 데이터 개수로 추정
                        frame_count = int(anim.samplers[0].output.count / 3)  # 각 샘플러의 출력 개수로 프레임 수 추정
                        print(f"시간 데이터가 없습니다. 출력 데이터에서 프레임 수를 추정: {frame_count}프레임")
                else:
                    frame_count = 0
                    print("애니메이션에 샘플러가 없습니다.")
                
                joint_positions = np.zeros((frame_count, len(smpl_humanml3d_to_mixamo_index), 3), dtype=np.float32)

                # 모든 시간값(타임스탬프) 수집
                for channel in anim.channels:
                    if channel.target.path == 'translation':  # 위치 애니메이션만 고려
                        sampler = anim.samplers[channel.sampler]
                        input_accessor = gltf.accessors[sampler.input]
                        time_data = get_accessor_data(gltf, input_accessor)
                        if time_data is not None:
                            time_values.extend(time_data.tolist())
                
                # 중복 제거 및 정렬
                time_values = sorted(set(time_values))
                
                # GLB 노드와 SMPL 본 간의 매핑 생성
                node_to_smpl_mapping = {}
                for i, node in enumerate(gltf.nodes):
                    node_name = getattr(node, 'name', f'node_{i}')
                    # 노드 이름을 SMPL 본 이름과 매칭
                    for smpl_idx, smpl_name in enumerate(smpl_humanml3d_to_mixamo_index):
                        # 이름 유사성으로 매칭 (대소문자 무시, 일부만 포함되어도 매칭)
                        if smpl_name.lower() == node_name.lower(): #  or node_name.lower() in smpl_name.lower():
                            node_to_smpl_mapping[i] = smpl_idx
                            print(f"매핑: GLB 노드 '{node_name}' -> SMPL 본 '{smpl_name}'")
                            break

                print(f"매핑된 본 수: {len(node_to_smpl_mapping)}/{len(smpl_humanml3d_to_mixamo_index)}")
    
                # 모든 시간값(타임스탬프) 수집
                for channel in anim.channels:
                    if channel.target.path == 'translation':  # 위치 애니메이션만 고려
                        sampler = anim.samplers[channel.sampler]
                        input_accessor = gltf.accessors[sampler.input]
                        time_data = get_accessor_data(gltf, input_accessor)
                        if time_data is not None:
                            time_values.extend(time_data.tolist())
                
                # 중복 제거 및 정렬
                time_values = sorted(set(time_values))
                
                # GLB 노드와 SMPL 본 간의 매핑 생성
                node_to_smpl_mapping = {}
                for i, node in enumerate(gltf.nodes):
                    node_name = getattr(node, 'name', f'node_{i}')
                    # 노드 이름을 SMPL 본 이름과 매칭
                    for smpl_idx, smpl_name in enumerate(smpl_humanml3d_to_mixamo_index):
                        # 이름 유사성으로 매칭 (대소문자 무시, 일부만 포함되어도 매칭)
                        if smpl_name.lower() == node_name.lower():
                            node_to_smpl_mapping[i] = smpl_idx
                            print(f"매핑: GLB 노드 '{node_name}' -> SMPL 본 '{smpl_name}'")
                            break
                
                print(f"매핑된 본 수: {len(node_to_smpl_mapping)}/{len(smpl_humanml3d_to_mixamo_index)}")
                
                # 각 타임스탬프에 대해 모든 관절 위치 계산
                for time_point in time_values:
                    # SMPL 본 구조에 맞는 배열 생성 (24개 본, 각각 [x,y,z] 위치)
                    smpl_frame_positions = np.zeros((len(smpl_humanml3d_to_mixamo_index), 3), dtype=np.float32)
                    
                    # 매핑된 GLB 노드에서 위치값 가져오기
                    for node_idx, smpl_idx in node_to_smpl_mapping.items():
                        position = calculate_node_position_at_time(gltf, node_idx, time_point)
                        smpl_frame_positions[smpl_idx] = position
                    
                    # 매핑되지 않은 본은 기본 위치 또는 인접한 본 위치 기반으로 추정
                    # (예: 부모 본 + 오프셋)
                    for smpl_idx, smpl_name in enumerate(smpl_humanml3d_to_mixamo_index):
                        if not any(v == smpl_idx for v in node_to_smpl_mapping.values()):
                            print(f"매핑되지 않은 SMPL 본: {smpl_name}, 기본값 사용")
                            # 기본값 사용 (0,0,0) - 이미 zeros로 초기화되어 있음
                            # 필요시 부모 본 기반으로 위치 추정 가능
                    
                    # Get current frame index
                    frame_idx = time_values.index(time_point)
                    # Assign joint positions to the pre-allocated array at the correct frame index
                    joint_positions[frame_idx] = smpl_frame_positions

                print(f"추출된 프레임 수: {frame_count}, 각 프레임당 본 수: {len(smpl_humanml3d_to_mixamo_index)}")
                return frame_count, joint_positions
            else:
                print(f"파일 {Path(file_path).name}에 애니메이션 정보가 없습니다.")
                return 0, None
        except Exception as e:
            print(f"GLB 파일에서 프레임 수 추출 오류: {e}")
            import traceback
            traceback.print_exc()
            return 0, None

    # 통합 학습 데이터(PT) 생성 함수 추가
    def generate_unified_training_file(directory):
        try:
            # GLB/FBX 파일 찾기
            files = find_org_files(directory)
            if not files:
                return "변환할 파일이 없습니다."
            
            # 출력 파일 경로 생성
            output_dir = Path(directory)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"mdm_dataset_{timestamp}.pt"
            
            # 모션 데이터 수집
            motions = []
            all_poses = []
            all_trans = []
            
            for file_path in files:
                file_ext = Path(file_path).suffix.lower()
                if file_ext not in ['.glb', '.fbx']:
                    continue
                
                # 설명 텍스트 가져오기
                txt_file_path = Path(file_path).with_suffix('.txt')
                if txt_file_path.exists():
                    with open(txt_file_path, 'r', encoding='utf-8') as f:
                        description = f.read().strip()
                else:
                    # 파일명에서 설명 생성
                    name_parts = re.split(r'[ _\-.]', Path(file_path).stem)
                    description = ', '.join([part for part in name_parts if part])
                    with open(txt_file_path, 'w', encoding='utf-8') as f:
                        f.write(text_parser.encode_tagged(description.strip()))
                try:
                    # 여기서 실제 GLB/FBX 변환 코드 구현 필요
                    # 임시 구현: 더미 데이터 생성

                    # GLB 파일에서 실제 translation 데이터를 추출
                    if file_ext == '.glb':
                        frame_count, joint_positions = get_glb_info(file_path)
                    else:
                        # FBX 또는 다른 형식은 더미 데이터 사용
                        frame_count = 1
                        joint_positions = np.zeros((frame_count, 22, 3), dtype=np.float32)  # [frame_count, 22 joints, 3 coordinates]

                    # 개별 모션 정보 저장
                    motion = {
                        'pose': joint_positions.tolist(),
                        'text': description
                    }
                    # 모션 데이터의 사이즈 계산
                    pose_size_bytes = joint_positions.size * joint_positions.itemsize
                    text_size_bytes = len(motion['text'].encode('utf-8')) if isinstance(motion['text'], str) else 0
                    pose_size_bytes += text_size_bytes
                    size_mb = pose_size_bytes / (1024 * 1024)
                    print(f"모션 데이터 크기: {pose_size_bytes} bytes, frame_count: {frame_count}, pose itemsize = {joint_positions.itemsize}")

                    motions.append(motion)
                    
                except Exception as e:
                    print(f"파일 처리 중 오류: {file_path} - {e}")
                    continue
            
            if not motions:
                return "처리할 수 있는 파일이 없습니다."
            
            dataset = {
                'motions': motions,
                'metadata': {
                    'file_count': len(motions),
                    'total_frames': sum(len(m) for m in motions),
                    'created_at': timestamp
                }
            }
            json_output_path = output_path.with_suffix('.json')
            with open(json_output_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(dataset, ensure_ascii=False, indent=4))

            print(f"PyTorch dataset saved to {output_path}")

            return f"통합 학습 파일이 생성되었습니다: {output_path}\n총 {len(motions)}개 애니메이션 데이터 포함"
        except Exception as e:
            print(f"통합 학습 파일 생성 중 오류: {e}")
            return f"통합 파일 생성 중 오류 발생: {str(e)}"

    # 데이터셋 디렉토리 선택 기능
    with gr.Row(2):
        with gr.Column():
            dataset_dir_input = gr.Textbox(
                label="데이터셋 디렉토리 경로",
                value=str(dataset_dir),
                interactive=True
            )
            refresh_btn = gr.Button("새로고침")
        with gr.Column(2):
            # 3D 모델 뷰어
            viewer = gr.HTML("""
            <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                        display: flex; justify-content: center; align-items: center; color: #ccc;">
                <div style="text-align: center;">
                    <h3>모델이 표시될 영역</h3>
                    <p>모션을 선택하세요.</p>
                </div>
            </div>
            """)
    
    # 설명 텍스트 편집 영역
    with gr.Row():
        new_description = gr.Textbox(
            label="새 설명 텍스트(한 줄 마다 애니메이션 구간에 대한 설명과 시간을 입력. 예) 앞으로 걸어가다 #0.0#0.5(줄바꿈)뒤로 돌아간다. #0.5#1.2)",
            placeholder="애니메이션에 대한 설명을 입력하세요...",
            lines=3,
            interactive=True
        )
    with gr.Row():
        update_btn = gr.Button("설명 업데이트", variant="primary")
        gen_npz_btn = gr.Button("학습 파일 생성", variant="secondary")
        gen_unified_btn = gr.Button("학습 파일 통합 생성", variant="secondary")

    # 데이터셋 목록 테이블
    dataset_table = gr.Dataframe(
        headers=["애니메이션 이름", "설명 텍스트", "데이터 크기", "파일 경로"],
        datatype=["str", "str", "str", "str"],
        label="학습 데이터셋 목록",
        value=update_dataset_list(str(dataset_dir)),
        interactive=False
    )
    
    # 선택된 항목 정보 표시
    with gr.Row():
        selected_file = gr.Textbox(label="선택된 파일 경로", interactive=False)
        selected_name = gr.Textbox(label="애니메이션 이름", interactive=False)
        
    # 계속 반복 여부 확인을 위한 체크박스 추가
    with gr.Row():
        continue_checkbox = gr.Checkbox(
            label="계속 반복하시겠습니까?",
            value=False,
            interactive=True
        )
    
    
    result_message = gr.Textbox(label="결과", interactive=False)
    
    # 이벤트 연결
    refresh_btn.click(
        update_dataset_list,
        inputs=[dataset_dir_input],
        outputs=[dataset_table]
    )
    
    # 테이블 선택 이벤트
    def select_file(evt: gr.SelectData, table_data):
        print(f"선택된 인덱스: {evt.index[0]}, 데이터: {table_data}")
        print(type(table_data))

        # DataFrame에서 행 인덱스 접근
        try:
            # DataFrame이면 iloc 사용
            if hasattr(table_data, 'iloc'):
                selected_row = table_data.iloc[evt.index[0]]
                print(f'선택된 데이터(DataFrame): {selected_row}')
            # 리스트면 직접 인덱싱
            else:
                selected_row = table_data[evt.index[0]]
                print(f'선택된 데이터(리스트): {selected_row}')
            
            # 인덱스가 아니라 열 이름으로 접근하거나, 위치로 접근
            if hasattr(table_data, 'iloc'):
                # DataFrame인 경우
                file_path = selected_row.iloc[3] if len(selected_row) > 3 else ""  # 파일 경로는 4번째 열
                file_name = selected_row.iloc[0] if len(selected_row) > 0 else ""  # 이름은 1번째 열
                file_desc = selected_row.iloc[1] if len(selected_row) > 1 else ""  # 설명은 2번째 열
            else:
                # 리스트인 경우
                file_path = selected_row[3] if len(selected_row) > 3 else ""
                file_name = selected_row[0] if len(selected_row) > 0 else ""
                file_desc = selected_row[1] if len(selected_row) > 1 else ""
                
            # 파일 확장자 확인 및 3D 모델 표시
            file_ext = Path(file_path).suffix.lower() if file_path else ""
            viewer_html = ""
            
            if file_ext == '.glb':
                # animation_tab.py의 process_animation와 유사하게 구현
                from util.file_utils import apply_animation  # 필요한 함수 import

                # 파일 경로를 name 속성이 있는 객체로 변환
                file_obj = FilePathWrapper(file_path)
                
                # 고정된 tpose.glb를 스킨으로 사용
                VIEWER_PATH = Path(__file__).parent.parent / "static" / "viewer.html"
                MODELS_DIR = Path(__file__).parent.parent / "static" / "models"
                dummy_path = MODELS_DIR / "tpose.glb"
                
                print(f"더미 경로: {dummy_path}")
                print(f"파일 경로: {file_path}")
                
                # apply_animation 호출하여 모델 표시
                if os.path.exists(dummy_path):
                        dummy_obj = FilePathWrapper(str(dummy_path))
                        viewer_html = apply_animation(dummy_obj, file_obj, VIEWER_PATH, MODELS_DIR)
                else:
                    viewer_html = f"""
                    <div style="width: 100%; height: 300px; background-color: #333; border-radius: 8px; 
                            display: flex; justify-content: center; align-items: center; color: #ff5555;">
                        <div style="text-align: center;">
                            <h3>오류</h3>
                            <p>스킨 모델(tpose.glb)을 찾을 수 없습니다.</p>
                            <p>경로: {dummy_path}</p>
                        </div>
                    </div>
                    """
            else:
                # GLB 파일이 아닌 경우 기본 메시지 표시
                viewer_html = f"""
                <div style="width: 100%; height: 300px; background-color: #333; border-radius: 8px; 
                        display: flex; justify-content: center; align-items: center; color: #ccc;">
                    <div style="text-align: center;">
                        <h3>3D 모델 미리보기</h3>
                        <p>GLB 파일만 미리보기가 가능합니다</p>
                        <p>선택된 파일: {file_name} ({file_ext})</p>
                    </div>
                </div>
                """
            
            return file_path, file_name, file_desc, viewer_html
        except Exception as e:
            print(f"테이블 데이터 처리 오류: {e}")
            error_html = f"""
            <div style="width: 100%; height: 300px; background-color: #333; border-radius: 8px; 
                    display: flex; justify-content: center; align-items: center; color: #ff5555;">
                <div style="text-align: center;">
                    <h3>오류 발생</h3>
                    <p>{str(e)}</p>
                </div>
            </div>
            """
            return "", "", "", error_html
    
    # select_file 함수의 outputs에 viewer 추가
    dataset_table.select(
        select_file,
        inputs=[dataset_table],
        outputs=[selected_file, selected_name, new_description, viewer]
    )
    
    # 버튼 클릭 이벤트 수정
    update_btn.click(
        update_and_refresh,
        inputs=[selected_file, new_description, dataset_dir_input, continue_checkbox],
        outputs=[result_message, dataset_table]
    )
    
    # 학습 파일(NPZ) 생성 버튼
    gen_npz_btn.click(
        generate_and_refresh,
        inputs=[selected_file, new_description, dataset_dir_input, continue_checkbox],
        outputs=[result_message, dataset_table]
    )
    
    # 통합 학습 파일(PT) 생성 버튼
    gen_unified_btn.click(
        generate_unified_training_file,
        inputs=[dataset_dir_input],
        outputs=[result_message]
    )
    
    return None

# 단독 실행 시 테스트
if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_dataset_create_tab(".")
    demo.launch()


def calculate_node_position_at_time(gltf, node_idx, time_point):
    """
    특정 시간에 해당 노드의 글로벌 위치를 계산합니다.
    
    Args:
        gltf: pygltflib로 로드된 GLB 파일
        node_idx: 위치를 계산할 노드의 인덱스
        time_point: 위치를 계산할 시간점(초)
    
    Returns:
        계산된 해당 시간에서의 노드 위치 (x, y, z)
    """
    try:
        node = gltf.nodes[node_idx]
        position = None
        
        # 기본 위치가 설정되어 있으면 사용
        if hasattr(node, 'translation') and node.translation:
            position = np.array(node.translation, dtype=np.float32)
        else:
            position = np.zeros(3, dtype=np.float32)
        
        # 애니메이션 채널에서 보간된 위치 찾기
        for anim_idx, anim in enumerate(gltf.animations):
            for channel_idx, channel in enumerate(anim.channels):
                # 현재 노드에 적용되는 채널인지 확인
                if channel.target.node == node_idx and channel.target.path == 'translation':
                    sampler = anim.samplers[channel.sampler]
                    
                    # 샘플러의 입력(시간) 및 출력(위치값) 접근을 위한 accessor 인덱스
                    input_accessor_idx = sampler.input
                    output_accessor_idx = sampler.output
                    
                    # accessor가 유효한지 확인
                    if (input_accessor_idx < len(gltf.accessors) and 
                        output_accessor_idx < len(gltf.accessors)):
                        
                        # 시간 데이터 접근
                        time_accessor = gltf.accessors[input_accessor_idx]
                        time_view = get_accessor_data(gltf, time_accessor)
                        
                        # 위치 데이터 접근
                        pos_accessor = gltf.accessors[output_accessor_idx]
                        pos_view = get_accessor_data(gltf, pos_accessor)
                        
                        if time_view is not None and pos_view is not None:
                            # 보간 방식 결정 (LINEAR 또는 STEP)
                            interpolation = sampler.interpolation if hasattr(sampler, 'interpolation') else 'LINEAR'
                            
                            # 값 보간
                            if interpolation == 'STEP':
                                # STEP 보간: 가장 가까운 이전 키프레임의 값 사용
                                idx = np.searchsorted(time_view, time_point, side='right') - 1
                                if idx >= 0 and idx < len(pos_view):
                                    position = np.array(pos_view[idx], dtype=np.float32)
                            else:
                                # LINEAR 보간: 두 키프레임 사이 선형 보간
                                idx = np.searchsorted(time_view, time_point, side='right')
                                
                                # 첫 프레임보다 이전이면 첫 프레임 값 사용
                                if idx == 0:
                                    position = np.array(pos_view[0], dtype=np.float32)
                                # 마지막 프레임 이후면 마지막 프레임 값 사용
                                elif idx >= len(time_view):
                                    position = np.array(pos_view[-1], dtype=np.float32)
                                # 두 프레임 사이면 보간
                                else:
                                    t0, t1 = time_view[idx-1], time_view[idx]
                                    p0, p1 = np.array(pos_view[idx-1]), np.array(pos_view[idx])
                                    
                                    # 보간 계수(0~1)
                                    alpha = (time_point - t0) / (t1 - t0) if t1 > t0 else 0
                                    position = p0 + alpha * (p1 - p0)
        
        # 부모 노드의 변환을 재귀적으로 적용하여 글로벌 위치 계산
        position = apply_parent_transforms(gltf, node_idx, position, time_point)
        
        return position.tolist()
    
    except Exception as e:
        print(f"위치 계산 중 오류 발생: {e}")
        return [0, 0, 0]  # 오류 시 기본 위치

def get_accessor_data(gltf, accessor):
    """
    GLTF accessor에서 실제 데이터를 추출합니다.
    
    Args:
        gltf: pygltflib로 로드된 GLB 파일
        accessor: 데이터에 접근하기 위한 accessor 객체
    
    Returns:
        accessor가 가리키는 데이터 배열, 또는 None(오류 시)
    """
    try:
        # 버퍼뷰 인덱스 확인
        buffer_view_idx = accessor.bufferView
        if buffer_view_idx is None or buffer_view_idx >= len(gltf.bufferViews):
            return None
            
        buffer_view = gltf.bufferViews[buffer_view_idx]
        
        # 버퍼 인덱스 확인
        buffer_idx = buffer_view.buffer
        if buffer_idx is None or buffer_idx >= len(gltf.buffers):
            return None
            
        # 버퍼 데이터 접근
        buffer_data = gltf.get_data_from_buffer_uri(gltf.buffers[buffer_idx].uri)
        if not buffer_data:
            # 내장 GLB 버퍼의 경우
            buffer_data = gltf.binary_blob()
        
        # 오프셋 및 길이 계산
        offset = buffer_view.byteOffset if hasattr(buffer_view, 'byteOffset') else 0
        acc_offset = accessor.byteOffset if hasattr(accessor, 'byteOffset') else 0
        total_offset = offset + acc_offset
        
        # 데이터 타입과 요소 크기 결정
        component_type = accessor.componentType  # 5126: FLOAT, 5125: UNSIGNED_INT 등
        type_str = accessor.type  # 'VEC3', 'SCALAR' 등
        
        # 타입별 numpy 데이터 타입 및 요소 수 결정
        type_map = {
            5120: np.int8,    # BYTE
            5121: np.uint8,   # UNSIGNED_BYTE
            5122: np.int16,   # SHORT
            5123: np.uint16,  # UNSIGNED_SHORT
            5125: np.uint32,  # UNSIGNED_INT
            5126: np.float32  # FLOAT
        }
        
        count_map = {
            'SCALAR': 1,
            'VEC2': 2,
            'VEC3': 3,
            'VEC4': 4,
            'MAT2': 4,
            'MAT3': 9,
            'MAT4': 16
        }
        
        if component_type not in type_map or type_str not in count_map:
            return None
            
        np_type = type_map[component_type]
        count = count_map[type_str]
        
        # 바이너리 데이터를 numpy 배열로 변환
        element_size = np.dtype(np_type).itemsize * count
        shape = (accessor.count,) if count == 1 else (accessor.count, count)
        
        # 데이터 버퍼에서 필요한 부분만 뽑아서 numpy 배열로 변환
        raw_data = buffer_data[total_offset:total_offset + accessor.count * element_size]
        
        # 1차원 배열로 먼저 변환 후 형태 변경
        flat_array = np.frombuffer(raw_data, dtype=np_type)
        
        # SCALAR인 경우는 그대로, 아니면 지정된 형태로 변환
        if count == 1:
            return flat_array
        else:
            return flat_array.reshape(shape)
    
    except Exception as e:
        print(f"데이터 추출 중 오류: {e}")
        return None

def apply_parent_transforms(gltf, node_idx, position, time_point):
    """
    부모 노드의 변환을 재귀적으로 적용하여 글로벌 위치를 계산합니다.
    
    Args:
        gltf: pygltflib로 로드된 GLB 파일
        node_idx: 위치를 계산할 노드의 인덱스
        position: 로컬 위치 (NumPy 배열)
        time_point: 위치를 계산할 시간점(초)
    
    Returns:
        글로벌 위치 (NumPy 배열)
    """
    try:
        # 부모 노드 찾기
        parent_idx = -1
        for i, node in enumerate(gltf.nodes):
            if hasattr(node, 'children') and node.children and node_idx in node.children:
                parent_idx = i
                break
        
        # 부모가 없으면 현재 위치 반환
        if parent_idx == -1:
            return position
        
        # 부모 노드의 변환(translation, rotation, scale) 적용
        parent = gltf.nodes[parent_idx]
        
        # 부모의 로컬 변환 가져오기
        parent_translation = np.array(parent.translation) if hasattr(parent, 'translation') and parent.translation else np.zeros(3)
        parent_rotation = np.array(parent.rotation) if hasattr(parent, 'rotation') and parent.rotation else np.array([0, 0, 0, 1])  # 쿼터니언
        parent_scale = np.array(parent.scale) if hasattr(parent, 'scale') and parent.scale else np.ones(3)
        
        # 부모 노드의 애니메이션 변환 적용 (필요시)
        # 이 부분은 더 복잡하므로 별도 함수로 구현 가능
        
        # 위치에 적용: 스케일 → 회전 → 이동 순서로 적용
        # 스케일 적용
        scaled_position = position * parent_scale
        
        # 회전 적용 (쿼터니언)
        from scipy.spatial.transform import Rotation as R
        if np.sum(parent_rotation) != 1:  # 단위 쿼터니언이 아닌 경우만 적용
            rot = R.from_quat(parent_rotation)
            rotated_position = rot.apply(scaled_position)
        else:
            rotated_position = scaled_position
            
        # 이동 적용
        global_position = rotated_position + parent_translation
        
        # 재귀적으로 상위 부모들의 변환도 적용
        return apply_parent_transforms(gltf, parent_idx, global_position, time_point)
        
    except Exception as e:
        print(f"부모 변환 적용 중 오류: {e}")
        return position