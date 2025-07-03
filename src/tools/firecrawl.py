"""
Firecrawl API를 사용한 웹 스크래핑 도구

이 모듈은 Enhanced Shopping Agent에서 사용할 Firecrawl 웹 스크래핑 도구를 제공합니다.
LangChain의 @tool 데코레이터를 사용하여 도구를 정의하고, 마크다운 형식의 구조화된 콘텐츠를 반환합니다.

주요 기능:
- 웹페이지 스크래핑 및 마크다운 변환
- 스마트한 제목 추출 알고리즘
- 콘텐츠 길이 제한 및 자동 트런케이션
- 에러 처리 및 안전한 응답 반환
- LangGraph astream_events와 호환되는 도구 인터페이스
"""

import os
import ssl
import urllib3
from typing import Dict, Any
from langchain_core.tools import tool
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# SSL 경고 비활성화 (외부 환경에서 SSL 인증서 문제 해결)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()


def extract_title_from_content(content: str) -> str:
    """
    스크래핑된 콘텐츠에서 적절한 제목을 추출합니다.
    
    다양한 형식의 콘텐츠에서 제목을 추출하는 스마트한 알고리즘을 사용합니다:
    1. 마크다운 헤더 (# 제목)
    2. HTML title 태그
    3. 적당한 길이의 첫 번째 텍스트 라인
    4. 콘텐츠 앞부분 50자 (fallback)
    
    Args:
        content (str): 분석할 콘텐츠
        
    Returns:
        str: 추출된 제목
        
    Example:
        >>> extract_title_from_content("# 쇼핑몰 상품\n상품 설명...")
        "쇼핑몰 상품"
    """
    if not content:
        return "제목 없음"
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        
        # 마크다운 헤더에서 제목 추출 (우선순위 1)
        if line.startswith('#'):
            return line.lstrip('#').strip()
            
        # HTML title 태그에서 제목 추출 (우선순위 2)
        elif line.startswith('<title>') and line.endswith('</title>'):
            return line.replace('<title>', '').replace('</title>', '').strip()
            
        # 적당한 길이의 텍스트 라인 (우선순위 3)
        elif line and 10 <= len(line) <= 100:
            return line
    
    # 모든 방법이 실패한 경우 첫 50자 사용 (fallback)
    return content[:50].replace('\n', ' ').strip() + "..." if len(content) > 50 else content


@tool
def firecrawl_scrape_tool(url: str, content_max_length: int = 10000) -> Dict[str, Any]:
    """
    Enhanced Shopping Agent용 Firecrawl 웹 스크래핑 도구
    
    이 도구는 LangGraph의 astream_events에서 on_tool_start/end 이벤트를 자동으로 발생시켜
    UI에서 도구 실행 상태를 실시간으로 추적할 수 있게 합니다.
    
    웹페이지를 스크래핑하여 마크다운 형식으로 변환하고, 제목 추출 및 콘텐츠 길이 제한을
    적용하여 Enhanced Shopping Agent에서 사용하기 적합한 형태로 가공합니다.
    
    Args:
        url (str): 스크래핑할 웹페이지 URL
        content_max_length (int): 콘텐츠 최대 길이 (기본값: 10000자)
            - 큰 페이지의 경우 처리 시간과 메모리 사용량 최적화
            - LLM 토큰 제한 고려
            
    Returns:
        Dict[str, Any]: 스크래핑 결과
            - content: 마크다운 형식의 스크래핑된 콘텐츠
            - title: 추출된 페이지 제목
            - url: 원본 URL
            - content_length: 원본 콘텐츠 길이
            - content_truncated: 콘텐츠 잘림 여부
            - success: 성공 여부
            - error: 에러 메시지 (실패 시에만)
            
    Example:
        >>> result = firecrawl_scrape_tool.invoke({
        ...     "url": "https://example.com/product",
        ...     "content_max_length": 5000
        ... })
        >>> print(result["title"])
        "Example Product Page"
        
    Note:
        - Firecrawl API 키가 FIRECRAWL_API_KEY 환경변수에 설정되어 있어야 합니다
        - 스크래핑 실패 시에도 안전한 응답을 반환하여 전체 플로우가 중단되지 않습니다
    """
    try:
        # API 키 확인
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return {
                "error": "FIRECRAWL_API_KEY 환경변수가 설정되지 않았습니다.",
                "url": url,
                "success": False
            }
        
        # Firecrawl 클라이언트 초기화 및 스크래핑 수행
        client = FirecrawlApp(api_key=api_key)
        scrape_result = client.scrape_url(url, formats=["markdown"])
        
        if scrape_result and scrape_result.success:
            # 원본 콘텐츠 추출
            content = scrape_result.markdown or ""
            
            # 콘텐츠 길이 제한 적용
            limited_content = content[:content_max_length] if len(content) > content_max_length else content
            
            # 스마트 제목 추출
            title = extract_title_from_content(limited_content)
            
            # 성공 응답 구성
            return {
                "content": limited_content,
                "title": title,
                "url": url,
                "content_length": len(content),
                "content_truncated": len(content) > content_max_length,
                "success": True
            }
        else:
            # Firecrawl API 레벨 실패
            error_msg = scrape_result.error if scrape_result else "Firecrawl API 응답 없음"
            return {
                "error": error_msg,
                "url": url,
                "success": False
            }
            
    except Exception as e:
        # 예외 발생 시 안전한 응답 반환
        return {
            "error": str(e),
            "url": url,
            "success": False
        }


if __name__ == "__main__":
    # 개발/테스트용 실행 코드
    test_url = "https://example.com"
    result = firecrawl_scrape_tool.invoke({
        "url": test_url,
        "content_max_length": 5000
    })
    print(f"스크래핑 테스트 결과: {result}")