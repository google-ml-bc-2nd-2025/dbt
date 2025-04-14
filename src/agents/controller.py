"""
컨트롤러 모듈
프론트엔드와 에이전트 시스템 간의 통신을 관리하고 작업 상태를 추적합니다.
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from .base import BaseAgent, Message
from ..utils.logger import get_agent_logger, get_system_logger

# 로거 설정
agent_logger = get_agent_logger()
system_logger = get_system_logger()

@dataclass
class TaskStatus:
    """작업 상태 정보"""
    task_id: str
    status: str  # 'pending', 'processing', 'completed', 'error'
    prompt: str
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0

class Controller(BaseAgent):
    def __init__(self, redis_client):
        """컨트롤러 초기화"""
        super().__init__("controller", redis_client)
        self.tasks: Dict[str, TaskStatus] = {}
        self.websocket = None
        agent_logger.info("Controller 초기화 완료")
        
    def set_websocket(self, websocket):
        """웹소켓 연결 설정"""
        self.websocket = websocket
        agent_logger.debug(f"WebSocket 연결 상태 변경: {'연결됨' if websocket else '연결 해제'}")
        
    async def process_message(self, message: Message):
        """메시지 처리"""
        agent_logger.info(f"메시지 수신: sender={message.sender}, intent={message.intent}")
        
        try:
            if message.intent == "analyze_request":
                await self._handle_analyze_request(message)
            elif message.intent == "analysis_result":
                await self._handle_analysis_result(message)
            elif message.intent == "execution_complete":
                await self._handle_execution_complete(message)
            elif message.intent == "execution_error":
                await self._handle_execution_error(message)
            elif message.intent == "progress_update":
                await self._handle_progress_update(message)
            else:
                agent_logger.warning(f"알 수 없는 메시지 intent: {message.intent}")
                
        except Exception as e:
            agent_logger.error(f"메시지 처리 중 오류 발생: {e}", exc_info=True)
            
    async def _handle_analyze_request(self, message: Message):
        """동작 분석 요청 처리"""
        try:
            prompt = message.content.get("prompt", "")
            task_id = self._generate_task_id()
            
            agent_logger.info(f"분석 요청 처리 시작: task_id={task_id}, prompt={prompt}")
            
            # 작업 상태 초기화
            self.tasks[task_id] = TaskStatus(
                task_id=task_id,
                status="pending",
                prompt=prompt,
                start_time=datetime.now()
            )
            
            # 프론트엔드에 상태 업데이트 전송
            await self._send_status_update(task_id)
            
            # PromptAnalyzer에게 분석 요청
            await self.send_message(
                "prompt_analyzer",
                "analyze_prompt",
                {
                    "prompt": prompt,
                    "task_id": task_id
                }
            )
            agent_logger.info(f"분석 요청 전달 완료: task_id={task_id}")
            
        except Exception as e:
            agent_logger.error(f"분석 요청 처리 중 오류 발생: {e}", exc_info=True)
            await self._handle_error(task_id, str(e))
            
    async def _handle_analysis_result(self, message: Message):
        """분석 결과 처리"""
        task_id = message.content.get("task_id")
        if not task_id or task_id not in self.tasks:
            agent_logger.error(f"알 수 없는 작업 ID: {task_id}")
            return
            
        agent_logger.info(f"분석 결과 수신: task_id={task_id}")
        
        task = self.tasks[task_id]
        task.status = "processing"
        task.progress = 0.3
        await self._send_status_update(task_id)
        
        if message.content.get("success"):
            # WorkflowPlanner에게 결과 전달
            await self.send_message(
                "workflow_planner",
                "plan_workflow",
                {
                    "analysis_result": message.content.get("result"),
                    "task_id": task_id,
                    "original_prompt": task.prompt
                }
            )
            agent_logger.info(f"워크플로우 계획 요청 전달: task_id={task_id}")
        else:
            error = message.content.get("error", "분석 실패")
            agent_logger.error(f"분석 실패: task_id={task_id}, error={error}")
            await self._handle_error(task_id, error)
            
    async def _handle_execution_complete(self, message: Message):
        """실행 완료 처리"""
        task_id = message.content.get("task_id")
        if not task_id or task_id not in self.tasks:
            agent_logger.error(f"알 수 없는 작업 ID: {task_id}")
            return
            
        agent_logger.info(f"작업 실행 완료: task_id={task_id}")
        
        task = self.tasks[task_id]
        task.status = "completed"
        task.end_time = datetime.now()
        task.result = {
            "output_path": message.content.get("output_path"),
            "metadata": message.content.get("metadata", {})
        }
        task.progress = 1.0
        
        await self._send_status_update(task_id)
        agent_logger.info(f"작업 완료 상태 업데이트: task_id={task_id}")
        
    async def _handle_execution_error(self, message: Message):
        """실행 오류 처리"""
        task_id = message.content.get("task_id")
        error = message.content.get("error", "알 수 없는 오류")
        agent_logger.error(f"작업 실행 오류: task_id={task_id}, error={error}")
        await self._handle_error(task_id, error)
        
    async def _handle_progress_update(self, message: Message):
        """진행 상황 업데이트 처리"""
        task_id = message.content.get("task_id")
        if not task_id or task_id not in self.tasks:
            return
            
        progress = message.content.get("progress", 0.0)
        self.tasks[task_id].progress = progress
        await self._send_status_update(task_id)
        agent_logger.debug(f"진행률 업데이트: task_id={task_id}, progress={progress:.1%}")
        
    async def _handle_error(self, task_id: str, error: str):
        """오류 처리"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = "error"
            task.error = error
            task.end_time = datetime.now()
            await self._send_status_update(task_id)
            agent_logger.error(f"작업 오류 처리: task_id={task_id}, error={error}")
            
    async def _send_status_update(self, task_id: str):
        """상태 업데이트를 프론트엔드로 전송"""
        if self.websocket and task_id in self.tasks:
            try:
                status_data = asdict(self.tasks[task_id])
                status_data["start_time"] = status_data["start_time"].isoformat()
                if status_data["end_time"]:
                    status_data["end_time"] = status_data["end_time"].isoformat()
                    
                await self.websocket.send_json({
                    "type": "status_update",
                    "data": status_data
                })
                agent_logger.debug(f"상태 업데이트 전송: task_id={task_id}, status={status_data['status']}")
            except Exception as e:
                agent_logger.error(f"상태 업데이트 전송 실패: {e}", exc_info=True)
                
    def _generate_task_id(self) -> str:
        """작업 ID 생성"""
        from uuid import uuid4
        return str(uuid4())
        
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        if task_id in self.tasks:
            return asdict(self.tasks[task_id])
        return None
        
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """모든 작업 상태 조회"""
        return [asdict(task) for task in self.tasks.values()] 