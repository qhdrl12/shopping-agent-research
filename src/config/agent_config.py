"""
Enhanced Shopping Agent 설정 파일
검색/스크래핑 개수 및 기타 설정 관리
"""

from dataclasses import dataclass
from typing import List


@dataclass
class SearchConfig:
    """검색 관련 설정"""
    # Tavily 검색 설정
    max_keywords_to_search: int = 2  # 검색할 최대 키워드 개수
    max_results_per_keyword: int = 2  # 키워드당 최대 검색 결과 개수
    total_max_search_results: int = 3  # 전체 최대 검색 결과 개수 (크레딧 절약)
    
    # 검색 품질 설정
    search_depth: str = "basic"  # "basic" 또는 "advanced"
    
    # 검색 쿼리 최적화
    add_shopping_keywords: bool = True  # "쇼핑", "구매" 등 키워드 자동 추가


@dataclass 
class ScrapingConfig:
    """스크래핑 관련 설정"""
    # Firecrawl 스크래핑 설정
    max_urls_to_scrape: int = 1  # 스크래핑할 최대 URL 개수 (크레딧 절약)
    max_concurrent_scraping: int = 1  # 동시 스크래핑 개수
    
    # URL 필터링 설정
    preferred_shopping_domains: List[str] = None
    content_max_length: int = 1500  # 스크래핑 콘텐츠 최대 길이
    
    # Firecrawl 파라미터 설정
    use_main_content_only: bool = True  # 메인 콘텐츠만 추출
    formats: List[str] = None  # 출력 형식 ['markdown', 'html']
    include_tags: List[str] = None  # 포함할 HTML 태그
    exclude_tags: List[str] = None  # 제외할 HTML 태그
    
    # 상품 정보 추출 설정
    extract_price: bool = True
    extract_title: bool = True
    extract_description: bool = True

    def __post_init__(self):
        if self.preferred_shopping_domains is None:
            self.preferred_shopping_domains = [
                'naver.com', 'coupang.com', 'gmarket.com', '11st.co.kr',
                'auction.co.kr', 'ssg.com', 'lotte.com', 'wemakeprice.com',
                'tmon.co.kr', 'interpark.com', 'yes24.com', 'aladin.co.kr',
                'musinsa.com', 'oliveyoung.co.kr', 'hmall.com'
            ]
        
        if self.formats is None:
            self.formats = ['markdown']  # 기본적으로 markdown만 사용
            
        if self.include_tags is None:
            self.include_tags = ['title', 'h1', 'h2', 'h3', 'p', 'span', 'div', 'strong', 'b']
            
        if self.exclude_tags is None:
            self.exclude_tags = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement']


@dataclass
class AgentConfig:
    """전체 에이전트 설정"""
    # 모델 설정
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    
    # 처리 시간 제한
    max_processing_time: int = 120  # 초
    
    # 서브 설정
    search: SearchConfig | None = None
    scraping: ScrapingConfig | None = None
    
    # 디버그 설정
    debug_mode: bool = False
    log_detailed_steps: bool = True

    def __post_init__(self):
        if self.search is None:
            self.search = SearchConfig()
        if self.scraping is None:
            self.scraping = ScrapingConfig()


# 기본 설정 인스턴스
DEFAULT_CONFIG = AgentConfig()

# 성능 테스트용 설정 (더 많은 검색/스크래핑)
PERFORMANCE_TEST_CONFIG = AgentConfig(
    search=SearchConfig(
        max_keywords_to_search=3,
        max_results_per_keyword=5,
        total_max_search_results=10,
        search_depth="advanced"
    ),
    scraping=ScrapingConfig(
        max_urls_to_scrape=5,
        max_concurrent_scraping=3,
        content_max_length=3000
    )
)

# 크레딧 절약 설정 (최소한의 검색/스크래핑)
CREDIT_SAVING_CONFIG = AgentConfig(
    search=SearchConfig(
        max_keywords_to_search=1,
        max_results_per_keyword=1,
        total_max_search_results=2,
        search_depth="basic"
    ),
    scraping=ScrapingConfig(
        max_urls_to_scrape=1,
        max_concurrent_scraping=1,
        content_max_length=1000
    )
)


def get_config(config_name: str = "default") -> AgentConfig:
    """설정 이름으로 설정 객체 반환"""
    configs = {
        "default": DEFAULT_CONFIG,
        "performance": PERFORMANCE_TEST_CONFIG,
        "credit_saving": CREDIT_SAVING_CONFIG
    }
    
    return configs.get(config_name, DEFAULT_CONFIG)