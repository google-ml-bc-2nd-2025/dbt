"""
워크플로우 플래너 모듈
동작 분석 결과를 기반으로 실제 모션 생성 워크플로우를 계획하고 실행합니다.
"""

from typing import Dict, Any
from .base import BaseAgent, Message

class WorkflowPlanner(BaseAgent):
    def __init__(self, redis_client):
        """워크플로우 플래너 초기화"""
        super().__init__("workflow_planner", redis_client)
        self.current_workflow = None
        
    async def process_message(self, message: Message):
        """메시지 처리"""
        if message.intent == "analysis_result":
            await self._handle_analysis_result(message)
            
    async def _handle_analysis_result(self, message: Message):
        """동작 분석 결과 처리"""
        try:
            content = message.content
            if not content.get("success"):
                await self._handle_analysis_error(content)
                return
                
            result = content.get("result", {})
            original_prompt = content.get("original_prompt", "")
            
            # 워크플로우 생성
            workflow = self._create_workflow(result, original_prompt)
            self.current_workflow = workflow
            
            # 태스크 실행기에게 워크플로우 전달
            await self.send_message(
                "task_executor",
                "execute_workflow",
                {
                    "workflow": workflow,
                    "original_prompt": original_prompt
                }
            )
            
        except Exception as e:
            print(f"워크플로우 생성 중 오류 발생: {e}")
            await self._handle_analysis_error({"error": str(e)})
            
    def _create_workflow(self, analysis_result: Dict[str, Any], original_prompt: str) -> Dict[str, Any]:
        """분석 결과를 기반으로 워크플로우 생성"""
        motion = analysis_result.get("motion", {})
        speed = analysis_result.get("speed", {})
        intensity = analysis_result.get("intensity", {})
        
        # 모션 파라미터 설정
        motion_params = {
            "type": motion.get("type", "walking"),
            "direction": motion.get("direction", "forward"),
            "path": motion.get("path", "straight"),
            "speed": {
                "value": float(speed.get("value", 5)) / 10.0,  # 0-1 범위로 정규화
                "variation": speed.get("variation", "constant")
            },
            "intensity": {
                "value": float(intensity.get("value", 5)) / 10.0,  # 0-1 범위로 정규화
                "effort": intensity.get("effort", "normal")
            }
        }
        
        # 워크플로우 단계 정의
        workflow = {
            "steps": [
                {
                    "type": "motion_generation",
                    "params": motion_params
                },
                {
                    "type": "post_processing",
                    "params": {
                        "smooth": True,
                        "loop": motion.get("path") == "circular",
                        "cleanup": True
                    }
                }
            ],
            "metadata": {
                "original_prompt": original_prompt,
                "motion_type": motion.get("type"),
                "description": motion.get("description", "")
            }
        }
        
        return workflow
        
    async def _handle_analysis_error(self, error_content: Dict[str, Any]):
        """분석 오류 처리"""
        error_msg = error_content.get("error", "알 수 없는 오류가 발생했습니다.")
        
        # 클라이언트에게 오류 전송
        await self.send_message(
            "controller",
            "workflow_error",
            {
                "error": error_msg,
                "stage": "analysis"
            }
        ) 