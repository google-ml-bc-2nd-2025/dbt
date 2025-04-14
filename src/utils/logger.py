"""
로깅 설정 모듈
시스템 전반의 로깅 설정을 관리합니다.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

# 로그 디렉토리 설정
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 로그 파일명 설정
current_time = datetime.now().strftime("%Y%m%d")
SYSTEM_LOG_FILE = LOG_DIR / f"system_{current_time}.log"
API_LOG_FILE = LOG_DIR / f"api_{current_time}.log"
AGENT_LOG_FILE = LOG_DIR / f"agent_{current_time}.log"

# 로그 포맷 설정
LOG_FORMAT = logging.Formatter(
    '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """로거 설정"""
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    handler.setFormatter(LOG_FORMAT)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # 콘솔 출력 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LOG_FORMAT)
    logger.addHandler(console_handler)

    return logger

# 시스템 로거 설정
system_logger = setup_logger('system', SYSTEM_LOG_FILE)

# API 로거 설정
api_logger = setup_logger('api', API_LOG_FILE)

# 에이전트 로거 설정
agent_logger = setup_logger('agent', AGENT_LOG_FILE)

def get_system_logger() -> logging.Logger:
    """시스템 로거 반환"""
    return system_logger

def get_api_logger() -> logging.Logger:
    """API 로거 반환"""
    return api_logger

def get_agent_logger() -> logging.Logger:
    """에이전트 로거 반환"""
    return agent_logger 