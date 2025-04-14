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

def create_edit_dataset_tab(models_dir):
    """
    학습 데이터 수정 탭 UI를 생성합니다.
    
    Args:
        models_dir: 모델 파일이 저장된 디렉토리 경로
    """
    # 데이터셋 디렉토리 경로 설정
    dataset_dir = Path(models_dir).parent.parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    # MotionClip NPZ 파일을 찾는 함수
    def find_npz_files(directory):
        return sorted(glob.glob(str(Path(directory) / "*_motionclip.npz")))
    
    # 파일 정보를 추출하는 함수
    def get_file_info(file_path):
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
            return {
                "path": file_path,
                "name": os.path.basename(file_path),
                "text": f"오류: {str(e)}",
                "shape": "오류"
            }
    
    # 데이터셋 목록 업데이트
    def update_dataset_list(directory):
        files = find_npz_files(directory)
        file_infos = [get_file_info(file) for file in files]
        
        # Gradio 테이블에 맞는 형식으로 변환
        rows = [[info["name"], info["text"], info["shape"], info["path"]] for info in file_infos]
        return rows
    
    # 설명 텍스트 업데이트 함수
    def update_description(file_path, new_description):
        try:
            if not file_path or not os.path.exists(file_path):
                return "파일을 선택해주세요."
                
            # 파일 로드
            data = np.load(file_path, allow_pickle=True)
            file_data = {k: data[k] for k in data.keys()}
            
            # 설명 텍스트 업데이트
            if new_description.strip():
                file_data['text'] = np.array([new_description.strip()], dtype=object)
                
                # 임시 파일로 저장
                temp_path = file_path + ".temp"
                np.savez(temp_path, **file_data)
                
                # 원본 파일 교체
                os.replace(temp_path, file_path)
                return f"설명이 업데이트 되었습니다: '{new_description}'"
            else:
                return "설명을 입력해주세요."
                
        except Exception as e:
            return f"오류 발생: {str(e)}"
    
    # 파일 삭제 함수
    def delete_file(file_path):
        try:
            if not file_path or not os.path.exists(file_path):
                return "파일을 선택해주세요."
                
            os.remove(file_path)
            return f"파일이 삭제되었습니다: {os.path.basename(file_path)}"
        except Exception as e:
            return f"파일 삭제 중 오류 발생: {str(e)}"
    
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
    
    with gr.Row():
        update_btn = gr.Button("설명 업데이트", variant="primary")
        delete_btn = gr.Button("파일 삭제", variant="stop")
    
    result_message = gr.Textbox(label="결과", interactive=False)
    
    # 이벤트 연결
    refresh_btn.click(
        update_dataset_list,
        inputs=[dataset_dir_input],
        outputs=[dataset_table]
    )
    
    # 테이블 선택 이벤트
    def select_file(evt: gr.SelectData, table_data):
        selected_row = table_data[evt.index[0]]
        file_path = selected_row[3]  # 파일 경로는 4번째 열
        file_name = selected_row[0]  # 이름은 1번째 열
        file_desc = selected_row[1]  # 설명은 2번째 열
        return file_path, file_name, file_desc
    
    dataset_table.select(
        select_file,
        inputs=[dataset_table],
        outputs=[selected_file, selected_name, new_description]
    )
    
    # 버튼 클릭 이벤트
    update_btn.click(
        update_description,
        inputs=[selected_file, new_description],
        outputs=[result_message]
    ).then(
        update_dataset_list,
        inputs=[dataset_dir_input],
        outputs=[dataset_table]
    )
    
    delete_btn.click(
        delete_file,
        inputs=[selected_file],
        outputs=[result_message]
    ).then(
        update_dataset_list,
        inputs=[dataset_dir_input],
        outputs=[dataset_table]
    ).then(
        lambda: ("", "", ""),
        outputs=[selected_file, selected_name, new_description]
    )
    
    return None

# 단독 실행 시 테스트
if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_edit_dataset_tab(".")
    demo.launch()
