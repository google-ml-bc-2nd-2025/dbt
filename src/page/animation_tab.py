"""
애니메이션 생성 탭 모듈
"""

import gradio as gr
import os
import uuid  # uuid 모듈 추가
from pathlib import Path
from util.file_utils import apply_animation, send_prompt
from render.smpl_animation import apply_to_glb
from render.humanml3d_renderer import render_humanml3d
import numpy as np
import base64
import json
from io import BytesIO
from pathlib import Path

def create_animation_tab(VIEWER_PATH, MODELS_DIR):
    """애니메이션 생성 탭 인터페이스 생성"""
    with gr.Column():
        gr.Markdown("# 애니메이션 생성")
        gr.Markdown("스킨이 있는 GLB 모델을 지정 후 원하는 애니메이션 데이터를 선택하세요.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 파일 업로드 영역
                skin_model = gr.File(
                    label="스킨 모델 (GLB)",
                    file_types=[".glb"],
                    type="file"
                )
                
                anim_model = gr.File(
                    label="애니메이션 파일 (GLB/BVH/NPY)",
                    file_types=[".glb", ".bvh", ".npy"],
                    type="file"
                )
                
                apply_btn = gr.Button("적용 및 보기", variant="primary")
                
                gr.Markdown("""
                ## 사용 방법
                1. 스킨이 있는 GLB 모델을 첫 번째 필드에 업로드합니다.
                2. 애니메이션 파일(GLB, BVH 또는 SMPL NPY)을 두 번째 필드에 업로드합니다.
                3. '적용 및 보기' 버튼을 클릭합니다.
                4. 오른쪽 패널에서 애니메이션이 적용된 모델을 확인합니다.
                
                **참고**: 
                - GLB 애니메이션: 두 모델의 리깅(뼈대) 구조가 동일해야 합니다. 즉시 확인 가능
                - BVH 애니메이션: 본 이름이 비슷하면 자동으로 매핑을 시도합니다. 본만 래더링 됨.
                - SMPL NPY 애니메이션: SMPL 호환 모델에 적용됩니다. 에러 발생하면 static/models에 생성된 anim_xx 파일을 선택 후 실행
                """)
                
            with gr.Column(scale=2):
                # 3D 모델 뷰어
                viewer = gr.HTML("""
                <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                         display: flex; justify-content: center; align-items: center; color: #ccc;">
                    <div style="text-align: center;">
                        <h3>모델이 표시될 영역</h3>
                        <p>모델을 업로드하고 '적용 및 보기' 버튼을 클릭하세요</p>
                    </div>
                </div>
                """)
                
                # 모델 뷰어 바로 아래에 프롬프트 입력 영역 배치
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        prompt_input = gr.Textbox(
                            label="프롬프트 입력",
                            placeholder="캐릭터에 적용할 프롬프트를 입력하세요...",
                            lines=3
                        )
                    
                    with gr.Column(scale=1, min_width=120):
                        prompt_btn = gr.Button("전송", variant="secondary")
                
                # 프롬프트 결과 출력 영역
                prompt_result = gr.HTML(visible=True)
        
        # 버튼 클릭 이벤트 처리를 위한 함수 정의
        def process_animation(skin, anim):
            import os
            from pathlib import Path

            # 파일 확장자 확인
            if anim is not None:
                file_ext = Path(anim.name).suffix.lower()
            else:
                file_ext = None

            # NPY 또는 NPZ 파일 처리 (SMPL 애니메이션)
            if file_ext in ['.npy', '.npz']:
                try:
                    return render_humanml3d(anim)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"SMPL 애니메이션 미리보기 오류: {str(e)}"
            
            # 기존 GLB/BVH 파일 처리
            print(f"skin = {skin}")
            print(f"anim = {anim}")
            return apply_animation(skin, anim, VIEWER_PATH, MODELS_DIR)
        
        # 버튼 클릭 이벤트 함수 등록
        apply_btn.click(
            fn=process_animation,
            inputs=[skin_model, anim_model],
            outputs=viewer
        )
        
        # 프롬프트 전송 버튼 클릭 이벤트
        prompt_btn.click(
            fn=send_prompt,
            inputs=prompt_input,
            outputs=prompt_result
        )