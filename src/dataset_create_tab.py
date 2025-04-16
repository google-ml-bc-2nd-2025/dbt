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
            text_str = text[0] if isinstance(text, np.ndarray) and text.size > 0 else "설명 없음"
            
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
                
            return {
                "path": file_path,
                "name": animation_name,
                "text": description_text,
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
                        f.write(new_description.strip())
                    
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
                pose = np.zeros((frame_count, 132), dtype=np.float32)  # [60, 132] 형태의 6D 포즈
                trans = np.zeros((frame_count, 3), dtype=np.float32)   # [60, 3] 형태의 위치
                betas = np.zeros(10, dtype=np.float32)                 # [10] 형태의 체형 파라미터
                
                # NPZ 파일 저장
                np.savez(
                    output_path,
                    pose=pose,
                    trans=trans,
                    betas=betas,
                    text=np.array([description.strip()], dtype=object)
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
            output_path = output_dir / f"motionclip_dataset_{timestamp}.pt"
            
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
                
                try:
                    # 여기서 실제 GLB/FBX 변환 코드 구현 필요
                    # 임시 구현: 더미 데이터 생성
                    frame_count = 60
                    pose = np.random.normal(0, 0.1, (frame_count, 132)).astype(np.float32)
                    trans = np.random.normal(0, 0.1, (frame_count, 3)).astype(np.float32)
                    
                    # 데이터 수집
                    all_poses.append(pose)
                    all_trans.append(trans)
                    
                    # 개별 모션 정보 저장
                    motion = {
                        'pose': torch.tensor(pose),
                        'trans': torch.tensor(trans),
                        'length': frame_count,
                        'text': description
                    }
                    motions.append(motion)
                    
                except Exception as e:
                    print(f"파일 처리 중 오류: {file_path} - {e}")
                    continue
            
            if not motions:
                return "처리할 수 있는 파일이 없습니다."
            
            # 표준화를 위한 통계 계산
            all_poses_np = np.concatenate(all_poses, axis=0)
            all_trans_np = np.concatenate(all_trans, axis=0)
            
            mean_pose = np.mean(all_poses_np, axis=0)
            std_pose = np.std(all_poses_np, axis=0)
            mean_trans = np.mean(all_trans_np, axis=0)
            std_trans = np.std(all_trans_np, axis=0)
            
            # MotionCLIP 포맷의 데이터셋 생성
            dataset = {
                'motions': motions,
                'mean': {
                    'pose': torch.tensor(mean_pose),
                    'trans': torch.tensor(mean_trans)
                },
                'std': {
                    'pose': torch.tensor(std_pose),
                    'trans': torch.tensor(std_trans)
                },
                'metadata': {
                    'file_count': len(motions),
                    'total_frames': sum(m['length'] for m in motions),
                    'created_at': timestamp
                }
            }
            
            # PyTorch 파일로 저장
            torch.save(dataset, output_path)
            
            return f"통합 학습 파일이 생성되었습니다: {output_path}\n총 {len(motions)}개 애니메이션 데이터 포함"
        except Exception as e:
            print(f"통합 학습 파일 생성 중 오류: {e}")
            return f"통합 파일 생성 중 오류 발생: {str(e)}"

    # 데이터셋 디렉토리 선택 기능
    with gr.Row():
        dataset_dir_input = gr.Textbox(
            label="데이터셋 디렉토리 경로",
            value=str(dataset_dir),
            interactive=True
        )
        refresh_btn = gr.Button("새로고침")
    
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
    
    # 설명 텍스트 편집 영역
    with gr.Row():
        new_description = gr.Textbox(
            label="새 설명 텍스트",
            placeholder="애니메이션에 대한 설명을 입력하세요...",
            lines=3,
            interactive=True
        )
    
    # 계속 반복 여부 확인을 위한 체크박스 추가
    with gr.Row():
        continue_checkbox = gr.Checkbox(
            label="계속 반복하시겠습니까?",
            value=False,
            interactive=True
        )
    
    with gr.Row():
        update_btn = gr.Button("설명 업데이트", variant="primary")
        gen_npz_btn = gr.Button("학습 파일 생성", variant="secondary")
        gen_unified_btn = gr.Button("학습 파일 통합 생성", variant="secondary")
    
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
                
            return file_path, file_name, file_desc
        except Exception as e:
            print(f"테이블 데이터 처리 오류: {e}")
            return "", "", ""
    
    dataset_table.select(
        select_file,
        inputs=[dataset_table],
        outputs=[selected_file, selected_name, new_description]
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
