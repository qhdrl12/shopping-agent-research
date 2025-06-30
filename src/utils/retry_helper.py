"""
간단한 재시도 헬퍼 유틸리티

API 호출 실패 시 재시도 로직을 제공합니다.
"""

import asyncio
import functools
from typing import Callable


def retry_on_failure(max_retries: int = 2, delay: float = 1.0):
    """
    API 호출 실패 시 재시도를 수행하는 데코레이터
    
    Args:
        max_retries (int): 최대 재시도 횟수 (기본값: 2)
        delay (float): 재시도 간 대기 시간 (초)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    print(f"⚠️ {func.__name__} 실패 (시도 {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"🔄 {delay}초 후 재시도...")
                    
                    await asyncio.sleep(delay)
            
            # 모든 재시도 실패 시 원본 예외 발생
            raise last_exception
        
        return wrapper
    return decorator