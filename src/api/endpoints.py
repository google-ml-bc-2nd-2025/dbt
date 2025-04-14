"""
FastAPI 엔드포인트 모듈
프론트엔드와의 HTTP/WebSocket 통신을 위한 엔드포인트를 정의합니다.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
import redis
import json
import time

from ..agents.controller import Controller
from ..agents.base import Message
from ..utils.logger import get_api_logger, get_system_logger

# 로거 설정
api_logger = get_api_logger()
system_logger = get_system_logger()

app = FastAPI(title="Motion Generator API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 origin으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis 클라이언트 설정
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

def get_redis_client():
    """Redis 클라이언트 생성"""
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        client.ping()  # 연결 테스트
        system_logger.info("Redis 연결 성공")
        return client
    except redis.ConnectionError as e:
        system_logger.error(f"Redis 연결 실패: {e}")
        raise HTTPException(status_code=500, detail="Redis 서버 연결 실패")

def get_controller():
    """Controller 인스턴스 생성"""
    redis_client = get_redis_client()
    return Controller(redis_client)

@app.get("/")
async def root():
    """API 상태 확인"""
    api_logger.info("API 상태 확인 요청")
    return {"status": "running", "service": "Motion Generator API"}

@app.post("/motion/generate")
async def generate_motion(
    prompt: str,
    controller: Controller = Depends(get_controller)
) -> Dict[str, Any]:
    """
    동작 생성 요청을 처리합니다.
    
    Args:
        prompt (str): 동작을 설명하는 자연어 프롬프트
        
    Returns:
        Dict[str, Any]: 작업 ID와 초기 상태 정보
    """
    start_time = time.time()
    api_logger.info(f"동작 생성 요청 수신: {prompt}")
    
    try:
        # 분석 요청 메시지 생성
        message = Message(
            sender="api",
            intent="analyze_request",
            content={"prompt": prompt}
        )
        
        # 컨트롤러에 메시지 전달
        await controller.process_message(message)
        
        # 생성된 작업의 ID와 상태 반환
        tasks = controller.get_all_tasks()
        if tasks:
            latest_task = tasks[-1]  # 가장 최근 작업
            processing_time = time.time() - start_time
            api_logger.info(
                f"동작 생성 요청 처리 완료: task_id={latest_task['task_id']}, "
                f"처리 시간={processing_time:.2f}초"
            )
            return {
                "task_id": latest_task["task_id"],
                "status": latest_task["status"],
                "message": "동작 생성 요청이 접수되었습니다."
            }
        else:
            api_logger.error("작업 생성 실패: 작업 목록이 비어있음")
            raise HTTPException(status_code=500, detail="작업 생성 실패")
            
    except Exception as e:
        api_logger.error(f"동작 생성 요청 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/motion/status/{task_id}")
async def get_motion_status(
    task_id: str,
    controller: Controller = Depends(get_controller)
) -> Dict[str, Any]:
    """
    특정 작업의 상태를 조회합니다.
    
    Args:
        task_id (str): 조회할 작업 ID
        
    Returns:
        Dict[str, Any]: 작업 상태 정보
    """
    api_logger.info(f"작업 상태 조회: task_id={task_id}")
    
    status = controller.get_task_status(task_id)
    if status is None:
        api_logger.warning(f"작업을 찾을 수 없음: task_id={task_id}")
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        
    api_logger.info(f"작업 상태 조회 완료: task_id={task_id}, status={status['status']}")
    return status

@app.get("/motion/tasks")
async def get_all_tasks(
    controller: Controller = Depends(get_controller)
) -> List[Dict[str, Any]]:
    """
    모든 작업 목록을 조회합니다.
    
    Returns:
        List[Dict[str, Any]]: 작업 목록
    """
    api_logger.info("전체 작업 목록 조회")
    tasks = controller.get_all_tasks()
    api_logger.info(f"전체 작업 목록 조회 완료: {len(tasks)}개 작업")
    return tasks

@app.websocket("/ws/motion/{task_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    task_id: str,
    controller: Controller = Depends(get_controller)
):
    """
    WebSocket 연결을 통해 실시간 작업 상태 업데이트를 제공합니다.
    
    Args:
        websocket (WebSocket): WebSocket 연결 객체
        task_id (str): 모니터링할 작업 ID
    """
    client_id = id(websocket)
    api_logger.info(f"WebSocket 연결 요청: client_id={client_id}, task_id={task_id}")
    
    await websocket.accept()
    controller.set_websocket(websocket)
    api_logger.info(f"WebSocket 연결 수락: client_id={client_id}")
    
    try:
        # 초기 상태 전송
        status = controller.get_task_status(task_id)
        if status:
            await websocket.send_json({
                "type": "status_update",
                "data": status
            })
            api_logger.debug(f"초기 상태 전송: client_id={client_id}, status={status['status']}")
            
        # WebSocket 연결 유지
        while True:
            # 클라이언트로부터의 메시지 대기
            data = await websocket.receive_text()
            
            # 필요한 경우 클라이언트 메시지 처리
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    api_logger.debug(f"Ping-Pong: client_id={client_id}")
            except json.JSONDecodeError:
                api_logger.warning(f"잘못된 WebSocket 메시지 형식: client_id={client_id}")
                
    except WebSocketDisconnect:
        api_logger.info(f"WebSocket 연결 종료: client_id={client_id}")
        # WebSocket 연결이 종료되면 컨트롤러에서 제거
        controller.set_websocket(None) 