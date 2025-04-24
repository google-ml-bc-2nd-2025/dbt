"""
애니메이션 생성 탭 모듈
"""

import gradio as gr
import os
import uuid  # uuid 모듈 추가
from pathlib import Path
from util.file_utils import apply_animation, send_prompt, save_model
from render.smpl_animation import apply_to_glb
import numpy as np
import base64
import json
from io import BytesIO
from pathlib import Path

def render_humanml3d(anim_file):
    import numpy as np
    import json
    import base64
    import uuid
    import os
    import shutil
    from pathlib import Path
    from util.file_utils import save_model
    
    print(f"[render_humanml3d] 애니메이션 파일 처리: {anim_file.name}")

    # 데이터 로드
    file_ext = Path(anim_file.name).suffix.lower()
    data = None

    if file_ext == '.npy':
        npy = np.load(anim_file.name, allow_pickle=True)
        print(f"npy= {type(npy)}, shape={npy.shape if hasattr(npy, 'shape') else 'None'}")
        
        # dict 타입 체크
        if isinstance(npy, dict) and 'motion' in npy:
            data = npy['motion']
        elif isinstance(npy, np.ndarray) and npy.dtype == np.dtype('O') and isinstance(npy.item(), dict):
            if 'motion' in npy.item():
                data = npy.item()['motion']
        else:
            # 일반 ndarray인 경우
            data = npy
            
    elif file_ext == '.npz':
        npz = np.load(anim_file.name, allow_pickle=True)
        # humanml3d 포맷에서 'motion' 또는 'poses' 키 사용
        if 'motion' in npz:
            data = npz['motion']
            print(f"npz['motion'] 데이터 로드: {data.shape if hasattr(data, 'shape') else 'None'}")
        elif 'poses' in npz:
            data = npz['poses']
            print(f"npz['poses'] 데이터 로드: {data.shape if hasattr(data, 'shape') else 'None'}")
        else:
            print(f"npz 키: {list(npz.keys())}")
            data = None
    
    if data is None:
        print("[render_humanml3d] 데이터를 읽을 수 없습니다.")
        return '<div>데이터를 읽을 수 없습니다.</div>'
    
    print(f"[render_humanml3d] 데이터 형태: {data.shape if hasattr(data, 'shape') else 'None'}")

    # (F, J, 3) 또는 (J, 3, F) 형태 지원
    if data.ndim == 4:
        print(f"[render_humanml3d] 4D 데이터 감지, 첫번째 시퀀스 사용: {data.shape}")
        data = data[0]
        
    if data.ndim == 3:
        print(f"[render_humanml3d] 3D 데이터 감지: {data.shape}")
        if data.shape[0] == 22 and data.shape[1] == 3:
            # (22, 3, F) -> (F, 22, 3)
            print(f"[render_humanml3d] (22, 3, F) 형태 감지, 변환 중")
            data = np.transpose(data, (2, 0, 1))
            print(f"[render_humanml3d] 변환 후 형태: {data.shape}")
        elif data.shape[1] == 22 and data.shape[2] == 3:
            # 이미 (F, 22, 3) 형태
            print(f"[render_humanml3d] (F, 22, 3) 형태 감지, 변환 불필요")
        else:
            print(f"[render_humanml3d] 지원하지 않는 데이터 형태: {data.shape}")
            return f'<div>지원하지 않는 데이터 형태입니다: {data.shape}</div>'
    else:
        print(f"[render_humanml3d] 지원하지 않는 데이터 차원: {data.ndim}")
        return f'<div>지원하지 않는 데이터 차원입니다: {data.ndim}</div>'

    # NaN/Inf 방지
    data = np.nan_to_num(data)

    # 임시 NPY 파일로 저장
    unique_id = uuid.uuid4().hex[:8]
    MODELS_DIR = Path(__file__).parent.parent / "static" / "models"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    temp_npy_path = MODELS_DIR / f"temp_humanml3d_{unique_id}.npy"

    # 임시 파일로 저장
    np.save(temp_npy_path, data)
    print(f"[render_humanml3d] 임시 파일 저장: {temp_npy_path}")
    
    # 가상 파일 객체 생성 (save_model 함수 요구사항)
    class MockFile:
        def __init__(self, path):
            self.name = path
    
    temp_file = MockFile(str(temp_npy_path))
    
    # viewer_template.html 파일 경로
    VIEWER_PATH = Path(__file__).parent.parent / "static" / "viewer_template.html"
    
    # GLB 방식과 동일하게 save_model 사용하여 anim_url 생성
    anim_url = save_model(temp_file, "anim", MODELS_DIR)
    
    # 임시 파일 자동 정리를 위한 스레드 시작
    import threading
    def cleanup_temp_file():
        import time
        time.sleep(300)  # 5분 후 정리
        try:
            if os.path.exists(temp_npy_path):
                os.remove(temp_npy_path)
                print(f"[render_humanml3d] 임시 파일 정리: {temp_npy_path}")
        except Exception as e:
            print(f"[render_humanml3d] 임시 파일 정리 실패: {e}")
    
    threading.Thread(target=cleanup_temp_file, daemon=True).start()
    
    # iframe으로 viewer_template.html 호출 (스킨 모델 없이 직접 호출)
    # animType=humanml3d 파라미터를 전달하여 HumanML3D 로더가 호출되도록 함
    viewer_url = f"/file={VIEWER_PATH}?anim={anim_url}&animType=humanml3d"
    print(f"[render_humanml3d] viewer URL: {viewer_url}")
    
    return f'''
    <div style="width: 100%; height: 500px; border-radius: 8px; overflow: hidden;">
        <iframe id="humanml3d-viewer-frame" src="{viewer_url}" style="width: 100%; height: 100%; border: none;"></iframe>
    </div>
    <p style="margin-top: 8px; color: #666; font-size: 0.9em;">
        마우스를 사용하여 모델을 회전하고 확대/축소할 수 있습니다.
    </p>
    '''

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