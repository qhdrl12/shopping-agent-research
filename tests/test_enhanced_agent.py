#!/usr/bin/env python3
"""
Enhanced Shopping Agent 테스트 스크립트
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 Python 패스에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

async def test_basic_query():
    """기본 쿼리 테스트"""
    print("=== Enhanced Shopping Agent 테스트 (크레딧 절약 모드) ===")
    
    # 설정 정보 출력
    from config.agent_config import get_config
    config = get_config("credit_saving")
    print(f"📊 설정 정보:")
    print(f"  - 최대 검색 키워드: {config.search.max_keywords_to_search}")
    print(f"  - 키워드당 최대 결과: {config.search.max_results_per_keyword}")
    print(f"  - 전체 최대 검색 결과: {config.search.total_max_search_results}")
    print(f"  - 최대 스크래핑 URL: {config.scraping.max_urls_to_scrape}")
    print()
    
    # 에이전트 인스턴스 생성 (크레딧 추적을 위해)
    from agent.enhanced_shopping_agent import EnhancedShoppingAgent
    agent_instance = EnhancedShoppingAgent(config)
    agent = agent_instance.create_workflow()
    
    # 테스트 쿼리 (크레딧 절약을 위해 1개만)
    test_queries = [
        "겨울용 패딩 점퍼 추천해줘. 10만원 이하로 검은색이면 좋겠어.",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"테스트 {i}: {query}")
        print('='*50)
        
        try:
            # 초기 상태 설정
            initial_state = {
                "user_query": query,
                "messages": [],
                "processing_status": "시작"
            }
            
            # 에이전트 실행 (스트림 모드)
            result = await agent.ainvoke(initial_state)
            
            # 결과 출력
            print(f"처리 상태: {result.get('processing_status', 'Unknown')}")
            
            if result.get('error_info'):
                print(f"❌ 오류: {result['error_info']}")
            
            # 단계별 결과 출력
            if result.get('analyzed_query'):
                print(f"\n📋 질문 분석:")
                analyzed = result['analyzed_query']
                print(f"  - 주요 상품: {analyzed.get('main_product', 'N/A')}")
                print(f"  - 검색 키워드: {analyzed.get('search_keywords', [])}")
                print(f"  - 가격대: {analyzed.get('price_range', 'N/A')}")
            
            if result.get('search_results'):
                print(f"\n🔍 검색 결과: {len(result['search_results'])}개")
                for i, result_item in enumerate(result['search_results'][:3], 1):
                    print(f"  {i}. {result_item.get('title', 'No title')}")
            
            if result.get('product_data'):
                print(f"\n🛍️ 수집된 상품: {len(result['product_data'])}개")
                for i, product in enumerate(result['product_data'][:3], 1):
                    print(f"  {i}. {product.get('name', 'No name')} - {product.get('price', 'No price')}")
            
            if result.get('final_answer'):
                print(f"\n💬 최종 답변:")
                print(result['final_answer'])
            
            # 처리 완료 알림
            print(f"\n✅ 테스트 완료")
            
        except Exception as e:
            print(f"❌ 테스트 실행 실패: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        
        # 테스트 간 대기
        await asyncio.sleep(2)


async def test_workflow_steps():
    """워크플로우 단계별 테스트"""
    print("\n=== 워크플로우 단계별 테스트 (크레딧 절약 모드) ===")
    
    from agent.enhanced_shopping_agent import EnhancedShoppingAgent
    from config.agent_config import get_config
    
    config = get_config("credit_saving")
    print(f"📊 Firecrawl 설정:")
    print(f"  - 출력 형식: {config.scraping.formats}")
    print(f"  - 메인 콘텐츠만: {config.scraping.use_main_content_only}")
    print(f"  - 포함 태그: {config.scraping.include_tags}")
    print(f"  - 제외 태그: {config.scraping.exclude_tags}")
    print()
    
    agent = EnhancedShoppingAgent(config)
    
    # 초기 상태
    test_state = {
        "user_query": "무선 이어폰 추천해줘. 5만원 이하로",
        "messages": [],
        "processing_status": "테스트"
    }
    
    try:
        # 1. 질문 분석 테스트
        print("\n1. 질문 분석 단계...")
        result = await agent.analyze_query(test_state)
        print(f"분석 결과: {result.get('analyzed_query', {})}")
        
        # 2. 사전 검색 테스트
        print("\n2. 사전 검색 단계...")
        result = await agent.pre_search(result)
        print(f"검색 결과 수: {len(result.get('search_results', []))}")
        
        # 3. 사전 스크래핑 테스트
        print("\n3. 사전 스크래핑 단계...")
        result = await agent.pre_scrape(result)
        print(f"스크래핑된 URL 수: {len(result.get('scraped_content', {}))}")
        
        # 4. React Agent 테스트
        print("\n4. React Agent 단계...")
        result = await agent.call_agent(result)
        print(f"최종 답변 길이: {len(result.get('final_answer', ''))}")
        
        # 5. 완료 알림
        print(f"\n✅ 모든 단계 완료")
        
    except Exception as e:
        print(f"❌ 단계별 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_firecrawl_direct():
    """Firecrawl 직접 클라이언트 테스트"""
    print("\n=== Firecrawl 직접 클라이언트 테스트 ===")
    
    try:
        from firecrawl import FirecrawlApp
        import os
        
        # Firecrawl 클라이언트 초기화
        firecrawl_client = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        
        # 테스트 URL (간단한 페이지)
        test_url = "https://example.com"
        
        print(f"🔍 테스트 URL: {test_url}")
        
        # 스크래핑 실행 (기본 파라미터만 사용)
        result = firecrawl_client.scrape_url(test_url)
        print(f"scrape_result: {result}")
        if result and result.success:
            content = result.markdown
            print(f"✅ 스크래핑 성공!")
            print(f"📄 콘텐츠 길이: {len(content)}자")
            print(f"📝 콘텐츠 미리보기:\n{content[:200]}...")
        else:
            error_msg = result.get("error") if result else "응답 없음"
            print(f"❌ 스크래핑 실패: {error_msg}")
            
    except Exception as e:
        print(f"❌ Firecrawl 테스트 실패: {str(e)}")


if __name__ == "__main__":
    async def main():
        await test_firecrawl_direct()
        await test_basic_query()
        await test_workflow_steps()
    
    asyncio.run(main())