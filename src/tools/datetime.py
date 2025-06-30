from datetime import datetime

from langchain.tools import tool

@tool
def get_current_time():
    """
    현재 시간을 조회할때 사용하는 함수 입니다.
    """
    return {
        "current_time": datetime.now(),
        "timezone": "Asia/Seoul"
    }
