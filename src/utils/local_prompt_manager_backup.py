import os
import json
from datetime import datetime
from typing import List, Dict, Optional

class LocalPromptManager:
    """로컬 파일 시스템을 사용하여 프롬프트를 관리하는 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: 프롬프트 데이터를 저장할 디렉토리 경로
        """
        self.data_dir = data_dir
        self.prompts_file = os.path.join(data_dir, "prompts.json")
        self._ensure_data_dir()
        self._ensure_prompts_file()
    
    def _ensure_data_dir(self):
        """데이터 디렉토리가 존재하지 않으면 생성"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _ensure_prompts_file(self):
        """프롬프트 파일이 존재하지 않으면 빈 파일 생성"""
        if not os.path.exists(self.prompts_file):
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def _load_prompts(self) -> List[Dict]:
        """프롬프트 데이터를 로드"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return []
    
    def _save_prompts(self, prompts: List[Dict]) -> bool:
        """프롬프트 데이터를 저장"""
        try:
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving prompts: {e}")
            return False
    
    def get_prompt_list(self) -> List[str]:
        """프롬프트 이름 목록을 가져옵니다."""
        try:
            prompts = self._load_prompts()
            return [prompt['name'] for prompt in prompts]
        except Exception as e:
            print(f"Error fetching prompt list: {e}")
            return []
    
    def get_prompt(self, name: str) -> Optional[Dict]:
        """이름으로 특정 프롬프트를 가져옵니다."""
        try:
            prompts = self._load_prompts()
            for prompt in prompts:
                if prompt['name'] == name:
                    return prompt
            return None
        except Exception as e:
            print(f"Error fetching prompt '{name}': {e}")
            return None
    
    def create_prompt(self, name: str, query_analysis_prompt: str, model_response_prompt: str) -> Optional[Dict]:
        """새 프롬프트를 생성합니다."""
        try:
            prompts = self._load_prompts()
            
            # 중복 이름 체크
            if any(prompt['name'] == name for prompt in prompts):
                print(f"Prompt with name '{name}' already exists.")
                return None
            
            # 새 ID 생성 (기존 최대 ID + 1)
            max_id = max([prompt.get('id', 0) for prompt in prompts], default=0)
            new_id = max_id + 1
            
            new_prompt = {
                'id': new_id,
                'name': name,
                'query_analysis_prompt': query_analysis_prompt,
                'model_response_prompt': model_response_prompt,
                'created_at': str(datetime.now()),
                'updated_at': str(datetime.now())
            }
            
            prompts.append(new_prompt)
            
            if self._save_prompts(prompts):
                return new_prompt
            return None
            
        except Exception as e:
            print(f"Error creating prompt: {e}")
            return None
    
    def update_prompt(self, prompt_id: int, name: str, query_analysis_prompt: str, model_response_prompt: str) -> Optional[Dict]:
        """기존 프롬프트를 수정합니다."""
        try:
            prompts = self._load_prompts()
            
            for i, prompt in enumerate(prompts):
                if prompt.get('id') == prompt_id:
                    prompts[i].update({
                        'name': name,
                        'query_analysis_prompt': query_analysis_prompt,
                        'model_response_prompt': model_response_prompt,
                        'updated_at': str(datetime.now())
                    })
                    
                    if self._save_prompts(prompts):
                        return prompts[i]
                    break
            
            print(f"Prompt with ID {prompt_id} not found.")
            return None
            
        except Exception as e:
            print(f"Error updating prompt: {e}")
            return None
    
    def delete_prompt(self, name: str) -> bool:
        """이름으로 프롬프트를 삭제합니다."""
        try:
            prompts = self._load_prompts()
            
            for i, prompt in enumerate(prompts):
                if prompt['name'] == name:
                    deleted_prompt = prompts.pop(i)
                    if self._save_prompts(prompts):
                        print(f"Prompt '{name}' deleted successfully.")
                        return True
                    break
            
            print(f"Prompt '{name}' not found.")
            return False
            
        except Exception as e:
            print(f"Error deleting prompt: {e}")
            return False

if __name__ == '__main__':
    from datetime import datetime
    
    # Example Usage
    manager = LocalPromptManager()
    
    # Get list of prompts
    print("Fetching prompt list...")
    prompt_list = manager.get_prompt_list()
    print(f"Available prompts: {prompt_list}")
    
    # Create a default prompt if none exists
    if not prompt_list:
        print("\nCreating default prompt...")
        default_prompt = manager.create_prompt(
            name="default",
            query_analysis_prompt="""당신은 친근하면서도 전문적인 쇼핑 컨설턴트입니다. 사용자가 입력한 쇼핑 관련 질문을 다각도로 분석하여, 실제 온라인 쇼핑 검색 및 상품 추천에 최적화된 구조화 정보를 추출해야 합니다. 아래의 단계별 지침과 예시, 제약조건을 반드시 준수하여 분석 결과를 JSON 형식으로 출력하세요.

---

**[분석 대상]**
- 사용자 질문: "{user_query}"

**[분석 및 추출 지침]**

1. **main_product (주요 상품)**
   - 사용자가 실제로 찾고자 하는 상품명 또는 가장 적합한 카테고리를 명확하게 한글로 기술하세요.
   - 상품명이 복수일 경우, 가장 핵심적인 상품 1개만 선정하세요.
   - 예시: "경량 패딩 점퍼", "블루투스 무선 이어폰", "런닝화"

2. **search_keywords (검색 키워드, 최대 5개, 중요도 순)**
   - 아래 유형별로 키워드를 추출하되, 실제 온라인 쇼핑몰에서 많이 사용되는 형태로 작성하세요.
   - [필수] 핵심 상품명 (예: "패딩", "이어폰", "런닝화")
   - [선택] 구체적 특징 (예: "경량", "방수", "노이즈캔슬링")
   - [선택] 브랜드명 (질문에 언급된 경우만)
   - [선택] 용도/시즌 (예: "겨울용", "운동용", "여름")
   - [선택] 성별/연령 (예: "남성용", "여성", "아동")
   - [선택] 가격대 키워드 (예: "저렴한", "가성비", "프리미엄")
   - [선택] 트렌드/스타일 (예: "트렌디한", "베이직", "클래식")
   - **중복/불필요한 키워드 제외, 너무 일반적이거나 너무 구체적인 키워드는 피하세요.**
   - **키워드는 한글로, 띄어쓰기 없이 작성하며, 실제 검색어처럼 자연스럽게 배열하세요.**
   - **예시**
     - "겨울에 입을 저렴한 남성 패딩 추천해줘" → ["남성패딩", "겨울패딩", "저렴한패딩", "경량패딩", "아우터"]
     - "여름용 여성 런닝화 비교" → ["여성런닝화", "여름운동화", "통기성운동화", "러닝화", "가벼운운동화"]

3. **price_range (가격대)**
   - 구체적 금액이 언급된 경우: "10만원 이하", "50-100만원"
   - 추상적 표현(저렴한, 가성비, 프리미엄 등)은 그대로 표기
   - 가격 관련 언급이 전혀 없으면 "가격 정보 없음"으로 명확히 작성
   - 예시: "저렴한", "20만원 이하", "가격 정보 없음"

4. **target_categories (대상 카테고리)**
   - 주 카테고리와 필요시 서브 카테고리를 모두 포함
   - 아래 표준 카테고리 중에서 선택(복수 가능):
     - 패션/의류, 전자제품, 생활용품, 스포츠/레저, 뷰티, 가전, 자동차, 도서, 식품, 유아동, 반려동물, 기타
   - 필요시 세부 카테고리 추가(예: "패션/의류 > 아우터", "전자제품 > 오디오")
   - 예시: ["패션/의류 > 아우터"], ["전자제품 > 오디오"]

5. **search_intent (검색 의도)**
   - 아래 중 하나로만 분류:
     - "구매" (즉시 구매 목적)
     - "비교" (여러 상품 비교 목적)
     - "정보수집" (상품 정보 탐색 목적)
     - "추천" (추천 요청 목적)
   - 질문에 명확한 의도가 없을 경우, 가장 근접한 의도를 논리적으로 추론
   - 예시: "추천해줘" → "추천", "어떤 게 좋은지 알려줘" → "비교"

**[에러 방지 및 검증 로직]**
- 모든 항목은 반드시 누락 없이 채워야 하며, 추정이 필요한 경우 논리적 근거에 따라 작성
- 키워드, 카테고리 등은 실제 쇼핑몰 검색에 적합한 형태로 작성
- 각 항목별로 예시와 가이드라인을 참고해 일관성 있게 작성
- 불명확하거나 모호한 질문의 경우, 최대한 명확하게 추론하되, 추론 근거를 간단히 주석으로 남김(출력에는 포함하지 않음)

**[출력 예시(JSON)]**
{
  "main_product": "경량 패딩 점퍼",
  "search_keywords": ["경량패딩", "남성패딩", "겨울패딩", "저렴한패딩", "아우터"],
  "price_range": "10만원 이하",
  "target_categories": ["패션/의류 > 아우터"],
  "search_intent": "구매"
}

---

위 지침을 반드시 준수하여, 사용자의 질문을 친근하지만 핵심을 정확히 파악하는 컨설턴트처럼 분석하고, 위 JSON 템플릿에 맞춰 구조화된 결과만 출력하세요.""",
            model_response_prompt="""**당신은 전문 쇼핑 컨설턴트입니다.**

**역할**: 사용자에게 최고의 쇼핑 경험을 제공하는 것이 목표입니다. 단순한 상품 나열이 아닌, 개인화된 맞춤 추천을 통해 사용자가 만족할 수 있는 완벽한 답변을 제공하세요.

**🎯 답변 구성 원칙**:

**1. 개인화된 인사 및 요구사항 확인**
- 사용자의 질문을 정확히 이해했음을 보여주세요
- 분석된 요구사항을 재확인하며 공감대 형성

**2. 핵심 추천 상품**
각 상품마다 다음 정보를 체계적으로 제공:
- **상품명**: 명확하고 구체적인 제품명
- **핵심 특징**: 왜 이 상품을 추천하는지 명확한 이유
- **가격 정보**: 구체적인 상품 가격
- **장점**: 사용자 요구사항과 연결된 장점
- **주의사항**: 솔직한 단점이나 고려사항 (신뢰성 향상)
- **구매처**: 구체적인 온라인몰이나 구매 방법

**3. 가격대별 세분화 추천**
- **경제적 선택**: 가성비 중심 옵션
- **균형 선택**: 가격과 품질의 균형
- **프리미엄 선택**: 최고 품질/성능 중심

**4. 실용적 구매 가이드**
- **구매 시 체크포인트**: 사이즈, 색상, 배송, A/S 등
- **계절성/시기 고려사항**: 언제 사는 것이 유리한지
- **대안 상품**: 재고 부족이나 예산 초과 시 대체재

**5. 전문가 팁 & 개인화 조언**
- 해당 카테고리의 전문적 인사이트
- 사용자 상황에 맞는 맞춤 조언
- 향후 구매를 위한 트렌드 정보

**🎨 답변 스타일 가이드**:
- **친근하고 전문적**: 딱딱하지 않으면서도 신뢰할 수 있는 톤
- **구체적이고 실용적**: 모호한 표현보다는 명확한 정보
- **균형잡힌 시각**: 장점만이 아닌 솔직한 단점도 포함
- **행동 유도**: 사용자가 다음에 무엇을 해야 할지 명확히 제시

**⚠️ 주의사항**:
- 수집된 정보가 부족한 경우, 솔직하게 한계를 인정하세요
- 과장된 표현보다는 객관적 정보를 우선하세요
- 가격은 변동 가능함을 명시하세요
- 개인 취향과 상황에 따라 다를 수 있음을 안내하세요

**📊 수집된 컨텍스트 정보**:
{context}

위 정보를 바탕으로 사용자에게 최고의 쇼핑 경험을 선사하는 완벽한 답변을 작성해주세요."""
        )
        if default_prompt:
            print(f"Created default prompt with ID: {default_prompt['id']}")
    
    # Get a specific prompt
    print("\nFetching 'default' prompt...")
    prompt = manager.get_prompt('default')
    if prompt:
        print(f"Fetched prompt: {prompt['name']}")
        print(f"ID: {prompt['id']}")
        print(f"Query Analysis: {prompt['query_analysis_prompt'][:100]}...")
        print(f"Model Response: {prompt['model_response_prompt'][:100]}...")
    
    print("\nLocal prompt management setup complete.")