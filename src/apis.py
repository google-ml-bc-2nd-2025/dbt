"""
3D 모델 뷰어 API 서버 모듈
FastAPI 기반의 RESTful API를 제공합니다.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio
import uvicorn
import os

# 사용자 정의 모듈 임포트
from file_utils import send_prompt

# 여기에 STATIC_DIR 정의
try:
    # app.py에서 STATIC_DIR을 가져오려고 시도
    from app import STATIC_DIR
except ImportError:
    # 가져오기 실패 시 여기서 직접 정의
    STATIC_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "static"

# FastAPI 애플리케이션 생성
app = FastAPI(title="3D 모델 뷰어 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (필요에 따라 제한)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 정적 파일 서빙을 위한 커스텀 StaticFiles 클래스
class CustomStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers):
        # glb 파일의 경우 항상 다시 로드하도록 수정
        return False

# 정적 파일 디렉토리 설정
app.mount("/static", CustomStaticFiles(directory=str(STATIC_DIR)), name="static")

# API 엔드포인트 정의
@app.get("/api/health")
async def health_check():
    """API 상태 확인을 위한 헬스 체크 엔드포인트"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/api/prompt")
async def process_prompt(request: Request):
    """프롬프트를 처리하는 API 엔드포인트"""
    data = await request.json()
    prompt = data.get("prompt", "")
    
    # 프롬프트 처리 로직
    try:
        result = send_prompt(prompt)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# 3D 모델 관련 API 엔드포인트
@app.get("/api/models")
async def list_models():
    """사용 가능한 3D 모델 목록 조회"""
    models_dir = STATIC_DIR / "models"
    
    # GLB 파일 목록
    glb_files = list(models_dir.glob("*.glb"))
    # BVH 파일 목록
    bvh_files = list(models_dir.glob("*.bvh"))
    # NPY 파일 목록 (SMPL 애니메이션)
    npy_files = list(models_dir.glob("*.npy"))
    
    return {
        "glb_models": [f.name for f in glb_files],
        "bvh_animations": [f.name for f in bvh_files],
        "smpl_animations": [f.name for f in npy_files]
    }

# 서버 실행 함수
def start_api_server(port=8000):
    """API 서버를 지정된 포트에서 실행합니다."""
    uvicorn.run(app, host="0.0.0.0", port=port)

# 비동기 서버 실행 함수 (백그라운드에서 실행)
async def start_api_server_async(port=8000):
    """API 서버를 비동기로 실행합니다."""
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info"))
    await server.serve()

# 직접 실행 시 API 서버 시작
if __name__ == "__main__":
    start_api_server()