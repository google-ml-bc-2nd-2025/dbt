"""
프롬프트 분석기 테스트
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.prompt_analyzer import PromptAnalyzer

@pytest.mark.asyncio
async def test_prompt_analysis():
    # Redis 클라이언트 목업
    redis_mock = MagicMock()
    
    # 분석기 초기화
    analyzer = PromptAnalyzer(redis_mock)
    
    # 모델 응답 목업
    analyzer.model = AsyncMock()
    analyzer.tokenizer = MagicMock()
    
    # 테스트 케이스
    test_cases = [
        {
            "input": "느리게 걷기",
            "expected_motion": "walking",
            "expected_speed": "slow"
        },
        {
            "input": "빠르게 뛰기",
            "expected_motion": "running",
            "expected_speed": "fast"
        }
    ]
    
    for case in test_cases:
        # 메시지 목업
        message = MagicMock()
        message.intent = "analyze_prompt"
        message.content = {"prompt": case["input"]}
        
        # 분석 수행
        await analyzer.process_message(message)
        
        # send_message 호출 확인
        analyzer.send_message.assert_called()
        
        # 마지막 호출의 인자 확인
        call_args = analyzer.send_message.call_args
        assert call_args is not None
        
        # 결과 검증
        _, intent, content = call_args[0]
        assert intent == "analysis_result"
        assert content["success"] is True
        assert "result" in content
        
@pytest.mark.asyncio
async def test_invalid_prompt():
    # Redis 클라이언트 목업
    redis_mock = MagicMock()
    
    # 분석기 초기화
    analyzer = PromptAnalyzer(redis_mock)
    
    # 빈 프롬프트 테스트
    message = MagicMock()
    message.intent = "analyze_prompt"
    message.content = {"prompt": ""}
    
    await analyzer.process_message(message)
    
    # 에러 메시지 전송 확인
    analyzer.send_message.assert_called_once()
    call_args = analyzer.send_message.call_args
    assert call_args is not None
    
    _, intent, content = call_args[0]
    assert intent == "error"
    assert "error" in content 