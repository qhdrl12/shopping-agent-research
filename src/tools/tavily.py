"""
Tavily API를 사용한 웹 검색 도구

이 모듈은 Enhanced Shopping Agent에서 사용할 Tavily 웹 검색 도구를 제공합니다.
LangChain의 @tool 데코레이터를 사용하여 도구를 정의하고, 구조화된 검색 결과를 반환합니다.

주요 기능:
- 웹 검색 수행 및 결과 구조화
- 에러 처리 및 안전한 응답 반환
- LangGraph astream_events와 호환되는 도구 인터페이스
"""

import os
from typing import Dict, Any
from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def web_search(query: str) -> str:
    """
    기본 웹 검색 도구 (레거시 호환용)
    
    Args:
        query: 검색 쿼리
        
    Returns:
        포맷된 검색 결과 문자열
        
    Note:
        이 함수는 기존 호환성을 위해 유지됩니다.
        새로운 구현에서는 tavily_search_tool을 사용하세요.
    """
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    response = client.search(query=query)
    
    results = []
    for i, result in enumerate(response.get("results", []), 1):
        results.append(
            f"{i}. **Title:** {result.get('title')}\n"
            f"   **URL:** {result.get('url')}\n"
            f"   **Content:** {result.get('content')}\n"
        )
    formatted_response = f"Search results for '{query}':\n\n" + "\n".join(results)
    return formatted_response


@tool
def tavily_search_tool(query: str, search_depth: str = "basic", max_results: int = 5) -> Dict[str, Any]:
    """
    Enhanced Shopping Agent용 Tavily 웹 검색 도구
    
    이 도구는 LangGraph의 astream_events에서 on_tool_start/end 이벤트를 자동으로 발생시켜
    UI에서 도구 실행 상태를 실시간으로 추적할 수 있게 합니다.
    
    Args:
        query (str): 검색 쿼리
        search_depth (str): 검색 깊이 
            - "basic": 기본 검색 (빠름)
            - "advanced": 고급 검색 (상세함)
        max_results (int): 최대 결과 수 (1-20)
        
    Returns:
        Dict[str, Any]: 구조화된 검색 결과
            - results: 검색 결과 리스트
            - results_count: 결과 개수
            - query: 원본 쿼리
            - search_depth: 사용된 검색 깊이
            - success: 성공 여부
            - error: 에러 메시지 (실패 시에만)
            
    Example:
        >>> result = tavily_search_tool.invoke({
        ...     "query": "겨울 패딩 추천",
        ...     "search_depth": "basic",
        ...     "max_results": 5
        ... })
        >>> print(result["results_count"])
        5
    """
    try:
        # API 키 확인
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "error": "TAVILY_API_KEY 환경변수가 설정되지 않았습니다.",
                "query": query,
                "success": False
            }
        
        # Tavily 클라이언트 초기화 및 검색 수행
        client = TavilyClient(api_key)
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results
        )
        
        # 성공 응답 구성
        return {
            "results": response.get("results", []),
            "results_count": len(response.get("results", [])),
            "query": query,
            "search_depth": search_depth,
            "success": True
        }
        
    except Exception as e:
        # 에러 발생 시 안전한 응답 반환
        return {
            "error": str(e),
            "query": query,
            "success": False
        }


if __name__ == "__main__":
    # 테스트 코드
    test_result = tavily_search_tool.invoke({
        "query": "Python 프로그래밍 튜토리얼",
        "search_depth": "basic",
        "max_results": 3
    })
    print(f"검색 테스트 결과: {test_result}")

