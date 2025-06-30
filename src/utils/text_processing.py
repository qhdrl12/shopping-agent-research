"""
텍스트 처리 유틸리티 함수들

이 모듈은 쇼핑 에이전트에서 사용되는 텍스트 처리 관련 기능들을 제공합니다.
- 제목 추출
- 상품 정보 추출  
- 가격 패턴 매칭
- 텍스트 정제 및 요약
"""

import re
from typing import Optional, Dict, Any
from datetime import datetime


def extract_title_from_content(content: str) -> str:
    """
    마크다운 콘텐츠에서 제목을 추출합니다.
    
    Args:
        content (str): 분석할 텍스트 콘텐츠
        
    Returns:
        str: 추출된 제목 또는 기본값
        
    Note:
        - 10-100자 사이의 첫 번째 유의미한 라인을 제목으로 간주
        - 너무 짧거나 긴 라인은 제외
    """
    if not content:
        return "제목 없음"
        
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        # 적절한 길이의 라인을 제목으로 간주
        if line and 10 <= len(line) <= 100:
            # 마크다운 헤더 기호 제거
            cleaned_title = re.sub(r'^#+\s*', '', line)
            return cleaned_title
            
    return "제목 없음"


def extract_price_from_content(content: str) -> str:
    """
    텍스트 콘텐츠에서 가격 정보를 추출합니다.
    
    Args:
        content (str): 분석할 텍스트 콘텐츠
        
    Returns:
        str: 추출된 가격 정보 또는 "가격 정보 없음"
        
    Note:
        - 한국 원화 패턴을 우선적으로 검색
        - 쉼표가 포함된 숫자 형식 지원
    """
    if not content:
        return "가격 정보 없음"
        
    # 다양한 가격 패턴 정의 (우선순위 순)
    price_patterns = [
        r'(\d{1,3}(?:,\d{3})*)\s*원',  # 1,000원 형식
        r'₩\s*(\d{1,3}(?:,\d{3})*)',   # ₩1,000 형식
        r'(\d{1,3}(?:,\d{3})*)\s*KRW', # 1,000 KRW 형식
        r'(\d{1,3}(?:,\d{3})*)\s*만원', # 10만원 형식
        r'(\d{1,3}(?:,\d{3})*)\s*천원', # 5천원 형식
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, content)
        if match:
            price_number = match.group(1)
            # 패턴에 따라 적절한 단위 추가
            if '만원' in pattern:
                return f"{price_number}만원"
            elif '천원' in pattern:
                return f"{price_number}천원"
            else:
                return f"{price_number}원"
    
    return "가격 정보 없음"


def extract_product_info_from_content(content: str, url: str) -> Optional[Dict[str, Any]]:
    """
    텍스트 콘텐츠에서 상품 정보를 추출합니다.
    
    Args:
        content (str): 분석할 텍스트 콘텐츠
        url (str): 상품 페이지 URL
        
    Returns:
        Optional[Dict[str, Any]]: 추출된 상품 정보 또는 None
        
    Note:
        - 제목, 가격, 설명, URL, 추출 시간 등을 포함
        - 추출 실패 시 None 반환
    """
    try:
        # 제목과 가격 추출
        title = extract_title_from_content(content)
        price = extract_price_from_content(content)
        
        # 상품 설명 생성 (콘텐츠 앞부분 200자)
        description = content[:200] + "..." if len(content) > 200 else content
        
        # 기본 상품 정보 구성
        product_info = {
            "name": title,
            "price": price,
            "description": description,
            "url": url,
            "availability": "확인 필요",  # 실제 재고 정보는 별도 로직 필요
            "extracted_at": datetime.now().isoformat(),
            "content_length": len(content)
        }
        
        return product_info
        
    except Exception as e:
        print(f"⚠️ 상품 정보 추출 실패: {str(e)}")
        return None


def clean_and_limit_content(content: str, max_length: int = 5000) -> str:
    """
    콘텐츠를 정제하고 길이를 제한합니다.
    
    Args:
        content (str): 원본 콘텐츠
        max_length (int): 최대 허용 길이 (기본값: 5000자)
        
    Returns:
        str: 정제된 콘텐츠
        
    Note:
        - 불필요한 공백과 특수문자 제거
        - 지정된 길이로 잘라내기
        - 마크다운 포맷 보존
    """
    if not content:
        return ""
    
    # 과도한 공백 제거
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # 연속된 빈 줄 정리
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)            # 연속된 공백 정리
    
    # 길이 제한
    if len(cleaned) > max_length:
        # 단어 경계에서 자르기
        truncated = cleaned[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # 80% 이상 위치라면 단어 경계에서 자르기
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    return cleaned


def calculate_relevance_score(search_result: Dict[str, Any], keyword: str) -> float:
    """
    검색 결과의 관련성 점수를 계산합니다.
    
    Args:
        search_result (Dict[str, Any]): 검색 결과 객체
        keyword (str): 검색 키워드
        
    Returns:
        float: 관련성 점수 (0.0 ~ 1.0+)
        
    Note:
        - 기본 점수에 다양한 가산점 적용
        - 제목, 내용, 쇼핑 관련 키워드 등을 고려
    """
    # 기본 점수 (검색 엔진에서 제공)
    base_score = search_result.get("score", 0.0)
    
    # 제목과 내용 추출
    title = search_result.get("title", "").lower()
    content = search_result.get("content", "").lower()
    keyword_lower = keyword.lower()
    
    # 관련성 점수 계산
    relevance_score = base_score
    
    # 키워드가 제목에 포함되면 큰 가산점
    if keyword_lower in title:
        relevance_score += 0.3
        
    # 키워드가 내용에 포함되면 중간 가산점
    if keyword_lower in content:
        relevance_score += 0.1
        
    # 쇼핑 관련 키워드 확인
    shopping_keywords = [
        "구매", "쇼핑", "가격", "할인", "배송", "리뷰", 
        "추천", "상품", "제품", "브랜드", "모델"
    ]
    
    shopping_keyword_count = sum(1 for shop_keyword in shopping_keywords 
                                if shop_keyword in title or shop_keyword in content)
    
    # 쇼핑 관련 키워드마다 소폭 가산점
    relevance_score += shopping_keyword_count * 0.02
    
    # URL 패턴 확인 (쇼핑몰 도메인)
    url = search_result.get("url", "").lower()
    shopping_domains = [
        "coupang", "11st", "gmarket", "auction", "interpark", 
        "wemakeprice", "tmon", "naver", "shopping", "store"
    ]
    
    if any(domain in url for domain in shopping_domains):
        relevance_score += 0.1
    
    return relevance_score


def format_search_results_for_display(search_results: list, max_results: int = 5) -> str:
    """
    검색 결과를 사용자에게 표시하기 위한 형태로 포맷팅합니다.
    
    Args:
        search_results (list): 검색 결과 리스트
        max_results (int): 표시할 최대 결과 수
        
    Returns:
        str: 포맷팅된 검색 결과 텍스트
    """
    if not search_results:
        return "검색 결과가 없습니다."
    
    formatted_results = []
    
    for i, result in enumerate(search_results[:max_results], 1):
        title = result.get('title', '제목 없음')
        url = result.get('url', '')
        content = result.get('content', '')[:150] + "..." if len(result.get('content', '')) > 150 else result.get('content', '')
        score = result.get('relevance_score', 0)
        
        formatted_result = f"""
**{i}. {title}**
- URL: {url}
- 관련성: {score:.2f}
- 내용: {content}
        """.strip()
        
        formatted_results.append(formatted_result)
    
    return "\n\n".join(formatted_results)