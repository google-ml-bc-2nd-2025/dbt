"""
기본 에이전트 클래스 정의
모든 에이전트의 기본 기능을 제공합니다.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import json
from redis import Redis

class Message:
    def __init__(self, sender: str, intent: str, content: Dict[str, Any]):
        self.sender = sender
        self.intent = intent  # 'request', 'response', 'error'
        self.content = content
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'sender': self.sender,
            'intent': self.intent,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        msg = cls(
            sender=data['sender'],
            intent=data['intent'],
            content=data['content']
        )
        msg.timestamp = datetime.fromisoformat(data['timestamp'])
        return msg

class BaseAgent:
    def __init__(self, agent_id: str, redis_client: Redis):
        self.agent_id = agent_id
        self.redis = redis_client
        self.message_queue = f"agent:{agent_id}:messages"
        self.state_key = f"agent:{agent_id}:state"
        self.running = False
    
    async def start(self):
        """에이전트 시작"""
        self.running = True
        await self.load_state()
        asyncio.create_task(self.message_loop())
    
    async def stop(self):
        """에이전트 중지"""
        self.running = False
        await self.save_state()
    
    async def message_loop(self):
        """메시지 처리 루프"""
        while self.running:
            # Redis에서 메시지 가져오기
            raw_message = self.redis.blpop(self.message_queue, timeout=1)
            if raw_message:
                _, message_data = raw_message
                message = Message.from_dict(json.loads(message_data))
                await self.process_message(message)
    
    async def process_message(self, message: Message):
        """메시지 처리 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    async def send_message(self, target_agent: str, intent: str, content: Dict[str, Any]):
        """다른 에이전트에게 메시지 전송"""
        message = Message(
            sender=self.agent_id,
            intent=intent,
            content=content
        )
        target_queue = f"agent:{target_agent}:messages"
        self.redis.rpush(target_queue, json.dumps(message.to_dict()))
    
    async def save_state(self):
        """현재 상태 저장"""
        state = self.get_current_state()
        self.redis.set(self.state_key, json.dumps(state))
    
    async def load_state(self):
        """저장된 상태 로드"""
        state_data = self.redis.get(self.state_key)
        if state_data:
            self.restore_state(json.loads(state_data))
    
    def get_current_state(self) -> Dict[str, Any]:
        """현재 상태 반환 (하위 클래스에서 구현)"""
        return {}
    
    def restore_state(self, state: Dict[str, Any]):
        """상태 복원 (하위 클래스에서 구현)"""
        pass 