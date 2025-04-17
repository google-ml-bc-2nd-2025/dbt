"""
애니메이션 생성 탭 모듈
"""

import gradio as gr
from pathlib import Path
import json
from file_utils import apply_animation, send_prompt, poll_task_status
from smpl_animation import apply_to_glb

def format_prompt_result(result: dict) -> str:
    """프롬프트 처리 결과를 텍스트로 포맷팅"""
    if result.get("status") == "error":
        return f"오류 발생: {result.get('message', '알 수 없는 오류')}"
    
    formatted_result = []
    for key, value in result.items():
        formatted_result.append(f"{key}: {value}")
    
    return "\n".join(formatted_result)

def handle_prompt(prompt: str) -> tuple:
    """프롬프트 처리 및 결과 업데이트"""
    if not prompt.strip():
        return "대기 중", "오류: 프롬프트를 입력해주세요."
    
    # 프롬프트 전송
    result = send_prompt(prompt)
    
    if result.get("status") == "success":
        task_id = result.get("task_id")
        if task_id:
            # 상태 확인 시작
            def check_status():
                try:
                    status = poll_task_status(task_id)
                    if status:
                        return f"상태: {status.get('status', '알 수 없음')}", format_prompt_result(status)
                except Exception as e:
                    return f"오류: {str(e)}", format_prompt_result(result)
                return "프롬프트 처리 중...", format_prompt_result(result)
            
            import threading
            threading.Thread(
                target=lambda: check_status(),
                daemon=True
            ).start()
            
            return "프롬프트 처리 시작...", format_prompt_result(result)
    
    return "대기 중", format_prompt_result(result)

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
                
                # 상태와 결과 출력 영역
                status_text = gr.Textbox(
                    label="처리 상태",
                    value="대기 중",
                    interactive=False
                )
                result_text = gr.Textbox(
                    label="프롬프트 처리 결과",
                    interactive=False,
                    lines=5
                )
                
                # 상태 확인 버튼 추가
                check_status_btn = gr.Button("상태 확인", variant="secondary")
        
        # 버튼 클릭 이벤트 처리를 위한 함수 정의
        def process_animation(skin, anim):
            """애니메이션 파일 형식에 따라 적절한 처리 함수를 호출"""
            # 스킨이 없으면 오류 메시지 표시
            if skin is None:
                return """
                <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                         display: flex; justify-content: center; align-items: center; color: #ccc;">
                    <div style="text-align: center;">
                        <h3>오류</h3>
                        <p>스킨 모델을 업로드해야 합니다</p>
                    </div>
                </div>
                """
            
            # 애니메이션이 없으면 스킨만 표시 (T 포즈)
            if anim is None:
                # 기존 apply_animation 함수를 재사용하여 스킨만 표시 (두 번째 인자는 None)
                return apply_animation(skin, None, VIEWER_PATH, MODELS_DIR)
            
            file_ext = Path(anim.name).suffix.lower()
            
            if file_ext == '.npy':
                # NPY 파일 처리 (SMPL 애니메이션)
                try:
                    return apply_to_glb(skin, anim, VIEWER_PATH, MODELS_DIR)
                except Exception as e:
                    return f"""
                    <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                             display: flex; justify-content: center; align-items: center; color: #ccc;">
                        <div style="text-align: center;">
                            <h3>SMPL 애니메이션 적용 오류</h3>
                            <p>{str(e)}</p>
                        </div>
                    </div>
                    """
            else:
                # 기존 GLB/BVH 파일 처리
                return apply_animation(skin, anim, VIEWER_PATH, MODELS_DIR)
        
        # 버튼 클릭 이벤트 함수 등록
        apply_btn.click(
            fn=process_animation,
            inputs=[skin_model, anim_model],
            outputs=viewer
        )
        
        # 프롬프트 전송 버튼 클릭 이벤트
        prompt_btn.click(
            fn=handle_prompt,
            inputs=prompt_input,
            outputs=[status_text, result_text]
        )
        
        # 상태 확인 버튼 클릭 이벤트
        def check_current_status(status_text, result_text):
            if not result_text:
                return status_text, result_text
            try:
                result = json.loads(result_text)
                task_id = result.get("task_id")
                if task_id:
                    status = poll_task_status(task_id)
                    if status:
                        return f"상태: {status.get('status', '알 수 없음')}", format_prompt_result(status)
            except:
                pass
            return status_text, result_text
        
        check_status_btn.click(
            fn=check_current_status,
            inputs=[status_text, result_text],
            outputs=[status_text, result_text]
        )