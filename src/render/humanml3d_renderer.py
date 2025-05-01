
import numpy as np
import uuid
import os
from pathlib import Path
from util.model_utils import save_model
from util.file_utils import cleanup_temp_files

def render_humanml3d(anim_file):
    print(f"[render_humanml3d] 애니메이션 파일 처리: {anim_file.name}")

    # 데이터 로드
    file_ext = Path(anim_file.name).suffix.lower()
    # Remove the leading dot from the extension
    if file_ext.startswith('.'):
        file_ext = file_ext[1:]
    data = None

    if file_ext == 'npy':
        npy = np.load(anim_file.name, allow_pickle=True)
        print(f"npy= {type(npy)}, shape={npy.shape if hasattr(npy, 'shape') else 'None'}")
        # 디버깅을 위해 npy 데이터의 처음 20개 요소 출력
        try:
            if hasattr(npy, 'shape'):
                print("npy 첫 20개 요소:", npy.flatten()[:20] if npy.size > 0 else "빈 배열")
            elif isinstance(npy, dict):
                keys = list(npy.keys())[:20]
                print(f"npy 딕셔너리 키 (최대 20개): {keys}")
            else:
                print("npy 내용:", str(npy)[:200], "...")
        except Exception as e:
            print(f"npy 출력 중 오류 발생: {e}")
        # dict 타입 체크
        if isinstance(npy, dict) and 'motion' in npy:
            data = npy['motion']
        elif isinstance(npy, np.ndarray) and npy.dtype == np.dtype('O') and isinstance(npy.item(), dict):
            if 'motion' in npy.item():
                data = npy.item()['motion']
        else:
            # 일반 ndarray인 경우
            data = npy
            
    elif file_ext == 'npz':
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
    cleanup_temp_files(MODELS_DIR)  # 이전 임시 파일 정리

    # 임시 파일로 저장
    np.save(temp_npy_path, data)
    print(f"[render_humanml3d] 임시 파일 저장: {temp_npy_path}")
    
    # 가상 파일 객체 생성 (save_model 함수 요구사항)
    class MockFile:
        def __init__(self, path):
            self.name = path
    
    temp_file = MockFile(str(temp_npy_path))
    
    # viewer.html 파일 경로
    TEMPLATE_PATH = Path(__file__).parent.parent / "static" / "viewer.html"
    
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
    
    # iframe으로 viewer.html 호출 (스킨 모델 없이 직접 호출)
    # animType=humanml3d 파라미터를 전달하여 HumanML3D 로더가 호출되도록 함
    viewer_url = f"/file={TEMPLATE_PATH}?anim={anim_url}&animType=humanml3d"
    print(f"[render_humanml3d] viewer URL: {viewer_url}")
    
    return f'''
    <div style="width: 100%; height: 500px; border-radius: 8px; overflow: hidden;">
        <iframe id="humanml3d-viewer-frame" src="{viewer_url}" style="width: 100%; height: 100%; border: none;"></iframe>
    </div>
    <p style="margin-top: 8px; color: #666; font-size: 0.9em;">
        마우스를 사용하여 모델을 회전하고 확대/축소할 수 있습니다.
    </p>
    '''