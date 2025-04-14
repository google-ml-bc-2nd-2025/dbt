"""
태스크 실행기 모듈
워크플로우 플래너로부터 받은 작업을 실제로 실행합니다.
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from .base import BaseAgent, Message

class TaskExecutor(BaseAgent):
    def __init__(self, redis_client, output_dir: Optional[str] = None):
        """태스크 실행기 초기화"""
        super().__init__("task_executor", redis_client)
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_task = None
        
    async def process_message(self, message: Message):
        """메시지 처리"""
        if message.intent == "execute_workflow":
            await self._execute_workflow(message)
            
    async def _execute_workflow(self, message: Message):
        """워크플로우 실행"""
        try:
            workflow = message.content.get("workflow", {})
            original_prompt = message.content.get("original_prompt", "")
            
            # 워크플로우 단계별 실행
            result = None
            for step in workflow.get("steps", []):
                result = await self._execute_step(step, result)
                
            if result is not None:
                # 결과 저장
                output_path = self._save_result(result, workflow)
                
                # 컨트롤러에게 완료 메시지 전송
                await self.send_message(
                    "controller",
                    "execution_complete",
                    {
                        "success": True,
                        "output_path": str(output_path),
                        "metadata": workflow.get("metadata", {}),
                        "original_prompt": original_prompt
                    }
                )
            else:
                raise Exception("모션 생성 결과가 없습니다.")
                
        except Exception as e:
            print(f"워크플로우 실행 중 오류 발생: {e}")
            await self._handle_execution_error(str(e))
            
    async def _execute_step(self, step: Dict[str, Any], previous_result: Any = None) -> Any:
        """워크플로우 단계 실행"""
        step_type = step.get("type")
        params = step.get("params", {})
        
        if step_type == "motion_generation":
            return await self._generate_motion(params)
        elif step_type == "post_processing":
            return await self._post_process(previous_result, params)
        else:
            raise ValueError(f"알 수 없는 단계 유형: {step_type}")
            
    async def _generate_motion(self, params: Dict[str, Any]) -> np.ndarray:
        """모션 생성"""
        try:
            # 모션 타입에 따른 기본 동작 생성
            motion_type = params.get("type", "walking")
            frames = 60  # 기본 1초 분량 (60fps)
            
            # 임시 구현: 기본 포즈 시퀀스 생성
            # 실제 구현에서는 여기에 모션 생성 모델 연동
            motion = np.zeros((frames, 24, 3))  # (프레임 수, 관절 수, 3차원)
            
            # 모션 타입별 기본 동작 설정
            if motion_type == "walking":
                motion = self._create_walking_motion(params)
            elif motion_type == "running":
                motion = self._create_running_motion(params)
            else:
                motion = self._create_default_motion(params)
                
            return motion
            
        except Exception as e:
            raise Exception(f"모션 생성 중 오류 발생: {e}")
            
    async def _post_process(self, motion: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """후처리 작업 수행"""
        if motion is None:
            raise ValueError("후처리할 모션 데이터가 없습니다.")
            
        try:
            # 스무딩 적용
            if params.get("smooth", False):
                motion = self._apply_smoothing(motion)
                
            # 루프 처리
            if params.get("loop", False):
                motion = self._make_loopable(motion)
                
            return motion
            
        except Exception as e:
            raise Exception(f"후처리 중 오류 발생: {e}")
            
    def _create_walking_motion(self, params: Dict[str, Any]) -> np.ndarray:
        """걷기 모션 생성"""
        frames = 60
        motion = np.zeros((frames, 24, 3))
        
        # 속도와 강도 반영
        speed = params.get("speed", {}).get("value", 0.5)
        intensity = params.get("intensity", {}).get("value", 0.5)
        
        # TODO: 실제 걷기 모션 생성 로직 구현
        # 현재는 임시로 단순한 사인 곡선 움직임 생성
        t = np.linspace(0, 2*np.pi, frames)
        motion[:, 0, 0] = np.sin(t * speed) * intensity  # 루트 조인트 x축 움직임
        
        return motion
        
    def _create_running_motion(self, params: Dict[str, Any]) -> np.ndarray:
        """달리기 모션 생성"""
        frames = 60
        motion = np.zeros((frames, 24, 3))
        
        # TODO: 실제 달리기 모션 생성 로직 구현
        speed = params.get("speed", {}).get("value", 0.7)
        intensity = params.get("intensity", {}).get("value", 0.7)
        
        t = np.linspace(0, 4*np.pi, frames)
        motion[:, 0, 0] = np.sin(t * speed) * intensity * 1.5
        
        return motion
        
    def _create_default_motion(self, params: Dict[str, Any]) -> np.ndarray:
        """기본 모션 생성"""
        frames = 60
        return np.zeros((frames, 24, 3))
        
    def _apply_smoothing(self, motion: np.ndarray) -> np.ndarray:
        """모션 스무딩 적용"""
        # 간단한 이동 평균 필터 적용
        kernel_size = 3
        smoothed = np.zeros_like(motion)
        
        for i in range(motion.shape[0]):
            start = max(0, i - kernel_size // 2)
            end = min(motion.shape[0], i + kernel_size // 2 + 1)
            smoothed[i] = motion[start:end].mean(axis=0)
            
        return smoothed
        
    def _make_loopable(self, motion: np.ndarray) -> np.ndarray:
        """모션을 루프 가능하도록 처리"""
        # 첫 프레임과 마지막 프레임을 보간하여 부드러운 전환 생성
        blend_frames = 10
        blended = motion.copy()
        
        for i in range(blend_frames):
            alpha = i / blend_frames
            blended[i] = (1 - alpha) * motion[0] + alpha * motion[-1]
            blended[-(i+1)] = alpha * motion[0] + (1 - alpha) * motion[-1]
            
        return blended
        
    def _save_result(self, motion: np.ndarray, workflow: Dict[str, Any]) -> Path:
        """결과 저장"""
        motion_type = workflow.get("metadata", {}).get("motion_type", "default")
        filename = f"motion_{motion_type}_{self._generate_timestamp()}.npy"
        output_path = self.output_dir / filename
        
        np.save(output_path, motion)
        return output_path
        
    async def _handle_execution_error(self, error_msg: str):
        """실행 오류 처리"""
        await self.send_message(
            "controller",
            "execution_error",
            {
                "error": error_msg,
                "stage": "execution"
            }
        )
        
    def _generate_timestamp(self) -> str:
        """타임스탬프 생성"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S") 