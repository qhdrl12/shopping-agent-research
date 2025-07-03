import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Literal

class LocalPromptManager:
    """
    프롬프트를 타입별로 독립적으로 관리하는 클래스
    
    각 프롬프트는 타입(query_analysis, model_response)별로 별도 관리되며,
    사용자가 각 타입에서 독립적으로 프롬프트를 선택할 수 있습니다.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: 프롬프트 데이터를 저장할 디렉토리 경로
        """
        self.data_dir = data_dir
        self.prompts_file = os.path.join(data_dir, "prompts_separated.json")
        self._ensure_data_dir()
        self._ensure_prompts_file()
    
    def _ensure_data_dir(self):
        """데이터 디렉토리가 존재하지 않으면 생성"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _ensure_prompts_file(self):
        """프롬프트 파일이 존재하지 않으면 기본 구조로 생성"""
        if not os.path.exists(self.prompts_file):
            default_data = {
                "query_analysis": [],
                "model_response": []
            }
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2)
    
    def _load_prompts(self) -> Dict[str, List[Dict]]:
        """프롬프트 데이터를 로드"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 기본 구조 보장
                if "query_analysis" not in data:
                    data["query_analysis"] = []
                if "model_response" not in data:
                    data["model_response"] = []
                return data
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return {"query_analysis": [], "model_response": []}
    
    def _save_prompts(self, prompts: Dict[str, List[Dict]]) -> bool:
        """프롬프트 데이터를 저장"""
        try:
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving prompts: {e}")
            return False
    
    def get_prompt_list_by_type_internal(self, prompt_type: Literal["query_analysis", "model_response"]) -> List[str]:
        """특정 타입의 프롬프트 이름 목록을 가져옵니다."""
        try:
            prompts_data = self._load_prompts()
            return [prompt['name'] for prompt in prompts_data.get(prompt_type, [])]
        except Exception as e:
            print(f"Error fetching prompt list for type '{prompt_type}': {e}")
            return []
    
    def get_prompt_by_type_internal(self, name: str, prompt_type: Literal["query_analysis", "model_response"]) -> Optional[Dict]:
        """이름과 타입으로 특정 프롬프트를 가져옵니다."""
        try:
            prompts_data = self._load_prompts()
            for prompt in prompts_data.get(prompt_type, []):
                if prompt['name'] == name:
                    return prompt
            return None
        except Exception as e:
            print(f"Error fetching prompt '{name}' of type '{prompt_type}': {e}")
            return None
    
    def create_prompt_by_type_internal(self, name: str, content: str, prompt_type: Literal["query_analysis", "model_response"]) -> Optional[Dict]:
        """새 프롬프트를 생성합니다."""
        try:
            prompts_data = self._load_prompts()
            
            # 중복 이름 체크 (같은 타입 내에서)
            if any(prompt['name'] == name for prompt in prompts_data.get(prompt_type, [])):
                print(f"Prompt with name '{name}' already exists in type '{prompt_type}'.")
                return None
            
            # 새 ID 생성 (해당 타입에서 최대 ID + 1)
            max_id = max([prompt.get('id', 0) for prompt in prompts_data.get(prompt_type, [])], default=0)
            new_id = max_id + 1
            
            new_prompt = {
                'id': new_id,
                'name': name,
                'content': content,
                'type': prompt_type,
                'created_at': str(datetime.now()),
                'updated_at': str(datetime.now())
            }
            
            prompts_data[prompt_type].append(new_prompt)
            
            if self._save_prompts(prompts_data):
                return new_prompt
            return None
            
        except Exception as e:
            print(f"Error creating prompt: {e}")
            return None
    
    def update_prompt_by_type_internal(self, prompt_id: int, name: str, content: str, prompt_type: Literal["query_analysis", "model_response"]) -> Optional[Dict]:
        """기존 프롬프트를 수정합니다."""
        try:
            prompts_data = self._load_prompts()
            
            for i, prompt in enumerate(prompts_data.get(prompt_type, [])):
                if prompt.get('id') == prompt_id:
                    prompts_data[prompt_type][i].update({
                        'name': name,
                        'content': content,
                        'updated_at': str(datetime.now())
                    })
                    
                    if self._save_prompts(prompts_data):
                        return prompts_data[prompt_type][i]
                    break
            
            print(f"Prompt with ID {prompt_id} not found in type '{prompt_type}'.")
            return None
            
        except Exception as e:
            print(f"Error updating prompt: {e}")
            return None
    
    def delete_prompt_by_type_internal(self, name: str, prompt_type: Literal["query_analysis", "model_response"]) -> bool:
        """이름과 타입으로 프롬프트를 삭제합니다."""
        try:
            prompts_data = self._load_prompts()
            
            for i, prompt in enumerate(prompts_data.get(prompt_type, [])):
                if prompt['name'] == name:
                    deleted_prompt = prompts_data[prompt_type].pop(i)
                    if self._save_prompts(prompts_data):
                        print(f"Prompt '{name}' of type '{prompt_type}' deleted successfully.")
                        return True
                    break
            
            print(f"Prompt '{name}' not found in type '{prompt_type}'.")
            return False
            
        except Exception as e:
            print(f"Error deleting prompt: {e}")
            return False
    
    def migrate_from_old_format(self, old_prompts_file: str = None) -> bool:
        """
        기존 prompts.json 형식에서 새로운 분리된 형식으로 마이그레이션합니다.
        """
        try:
            if old_prompts_file is None:
                old_prompts_file = os.path.join(self.data_dir, "prompts.json")
            
            if not os.path.exists(old_prompts_file):
                print("Old prompts file not found. Nothing to migrate.")
                return True
            
            # 기존 데이터 로드
            with open(old_prompts_file, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            
            prompts_data = self._load_prompts()
            
            # 각 프롬프트를 분리하여 저장
            for old_prompt in old_data:
                name = old_prompt.get('name', 'unnamed')
                
                # Query Analysis 프롬프트 마이그레이션
                if old_prompt.get('query_analysis_prompt'):
                    query_analysis_prompt = {
                        'id': len(prompts_data['query_analysis']) + 1,
                        'name': f"{name}_query_analysis",
                        'content': old_prompt['query_analysis_prompt'],
                        'type': 'query_analysis',
                        'created_at': old_prompt.get('created_at', str(datetime.now())),
                        'updated_at': old_prompt.get('updated_at', str(datetime.now())),
                        'migrated_from': name
                    }
                    prompts_data['query_analysis'].append(query_analysis_prompt)
                
                # Model Response 프롬프트 마이그레이션
                if old_prompt.get('model_response_prompt'):
                    model_response_prompt = {
                        'id': len(prompts_data['model_response']) + 1,
                        'name': f"{name}_model_response",
                        'content': old_prompt['model_response_prompt'],
                        'type': 'model_response',
                        'created_at': old_prompt.get('created_at', str(datetime.now())),
                        'updated_at': old_prompt.get('updated_at', str(datetime.now())),
                        'migrated_from': name
                    }
                    prompts_data['model_response'].append(model_response_prompt)
            
            # 새로운 형식으로 저장
            if self._save_prompts(prompts_data):
                print(f"Successfully migrated {len(old_data)} prompt sets to separated format.")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error during migration: {e}")
            return False
    
    def get_combined_prompt_sets(self) -> List[Dict]:
        """
        기존 호환성을 위해 결합된 프롬프트 세트 목록을 반환합니다.
        (query_analysis와 model_response가 같은 base name을 가진 경우 결합)
        """
        try:
            prompts_data = self._load_prompts()
            combined_sets = {}
            
            # Query Analysis 프롬프트 처리
            for prompt in prompts_data.get('query_analysis', []):
                base_name = prompt['name'].replace('_query_analysis', '')
                if base_name not in combined_sets:
                    combined_sets[base_name] = {}
                combined_sets[base_name]['query_analysis_prompt'] = prompt['content']
                combined_sets[base_name]['name'] = base_name
            
            # Model Response 프롬프트 처리
            for prompt in prompts_data.get('model_response', []):
                base_name = prompt['name'].replace('_model_response', '')
                if base_name not in combined_sets:
                    combined_sets[base_name] = {}
                combined_sets[base_name]['model_response_prompt'] = prompt['content']
                combined_sets[base_name]['name'] = base_name
            
            return list(combined_sets.values())
            
        except Exception as e:
            print(f"Error getting combined prompt sets: {e}")
            return []
    
    # ==========================================
    # 레거시 호환성 메서드들 (기존 app.py 지원용)
    # ==========================================
    
    def get_prompt_list(self) -> List[str]:
        """레거시: 결합된 프롬프트 세트 이름 목록을 반환합니다."""
        combined_sets = self.get_combined_prompt_sets()
        return [prompt_set.get('name', 'unnamed') for prompt_set in combined_sets]
    
    def get_prompt(self, name: str) -> Optional[Dict]:
        """레거시: 결합된 프롬프트 세트를 반환합니다."""
        combined_sets = self.get_combined_prompt_sets()
        for prompt_set in combined_sets:
            if prompt_set.get('name') == name:
                return prompt_set
        return None
    
    def create_prompt(self, name: str, query_analysis_prompt: str, model_response_prompt: str) -> Optional[Dict]:
        """레거시: 결합된 프롬프트 세트를 생성합니다."""
        try:
            # Query Analysis 프롬프트 생성
            query_result = self.create_prompt_by_type_internal(
                name=f"{name}_query_analysis",
                content=query_analysis_prompt,
                prompt_type="query_analysis"
            )
            
            # Model Response 프롬프트 생성
            response_result = self.create_prompt_by_type_internal(
                name=f"{name}_model_response", 
                content=model_response_prompt,
                prompt_type="model_response"
            )
            
            if query_result and response_result:
                return {
                    'name': name,
                    'query_analysis_prompt': query_analysis_prompt,
                    'model_response_prompt': model_response_prompt,
                    'created_at': str(datetime.now()),
                    'updated_at': str(datetime.now())
                }
            return None
            
        except Exception as e:
            print(f"Error creating combined prompt: {e}")
            return None
    
    def update_prompt(self, prompt_id: int, name: str, query_analysis_prompt: str, model_response_prompt: str) -> Optional[Dict]:
        """레거시: 결합된 프롬프트 세트를 업데이트합니다."""
        try:
            # 기존 프롬프트 찾기 및 업데이트
            query_name = f"{name}_query_analysis"
            response_name = f"{name}_model_response"
            
            # Query Analysis 업데이트
            prompts_data = self._load_prompts()
            query_prompt = None
            response_prompt = None
            
            for prompt in prompts_data.get('query_analysis', []):
                if prompt['name'] == query_name:
                    query_prompt = prompt
                    break
            
            for prompt in prompts_data.get('model_response', []):
                if prompt['name'] == response_name:
                    response_prompt = prompt
                    break
            
            if query_prompt:
                self.update_prompt_by_type_internal(
                    prompt_id=query_prompt['id'],
                    name=query_name,
                    content=query_analysis_prompt,
                    prompt_type="query_analysis"
                )
            
            if response_prompt:
                self.update_prompt_by_type_internal(
                    prompt_id=response_prompt['id'],
                    name=response_name,
                    content=model_response_prompt,
                    prompt_type="model_response"
                )
            
            return {
                'name': name,
                'query_analysis_prompt': query_analysis_prompt,
                'model_response_prompt': model_response_prompt,
                'updated_at': str(datetime.now())
            }
            
        except Exception as e:
            print(f"Error updating combined prompt: {e}")
            return None
    
    def delete_prompt(self, name: str) -> bool:
        """레거시: 결합된 프롬프트 세트를 삭제합니다."""
        try:
            query_name = f"{name}_query_analysis"
            response_name = f"{name}_model_response"
            
            success1 = self.delete_prompt_by_type_internal(query_name, "query_analysis")
            success2 = self.delete_prompt_by_type_internal(response_name, "model_response")
            
            return success1 or success2  # 둘 중 하나라도 성공하면 성공
            
        except Exception as e:
            print(f"Error deleting combined prompt: {e}")
            return False
    
    # 새로운 타입별 메서드들 (앞으로 사용할 메서드들)
    def create_prompt_by_type(self, name: str, content: str, prompt_type: Literal["query_analysis", "model_response"]) -> Optional[Dict]:
        """타입별 프롬프트 생성 (새로운 API)"""
        return self.create_prompt_by_type_internal(name, content, prompt_type)
    
    def update_prompt_by_type(self, prompt_id: int, name: str, content: str, prompt_type: Literal["query_analysis", "model_response"]) -> Optional[Dict]:
        """타입별 프롬프트 업데이트 (새로운 API)"""
        return self.update_prompt_by_type_internal(prompt_id, name, content, prompt_type)
    
    def delete_prompt_by_type(self, name: str, prompt_type: Literal["query_analysis", "model_response"]) -> bool:
        """타입별 프롬프트 삭제 (새로운 API)"""
        return self.delete_prompt_by_type_internal(name, prompt_type)
    
    def get_prompt_list_by_type(self, prompt_type: Literal["query_analysis", "model_response"]) -> List[str]:
        """타입별 프롬프트 목록 (새로운 API)"""
        return self.get_prompt_list_by_type_internal(prompt_type)
    
    def get_prompt_by_type(self, name: str, prompt_type: Literal["query_analysis", "model_response"]) -> Optional[Dict]:
        """타입별 프롬프트 조회 (새로운 API)"""
        return self.get_prompt_by_type_internal(name, prompt_type)


if __name__ == '__main__':
    # 예제 사용법
    manager = LocalPromptManager()
    
    print("=== 분리된 프롬프트 관리자 테스트 ===")
    
    # 기존 데이터 마이그레이션 (있을 경우)
    print("\n1. 기존 데이터 마이그레이션...")
    manager.migrate_from_old_format()
    
    # Query Analysis 프롬프트 목록
    print("\n2. Query Analysis 프롬프트 목록:")
    query_analysis_list = manager.get_prompt_list("query_analysis")
    print(f"Available: {query_analysis_list}")
    
    # Model Response 프롬프트 목록
    print("\n3. Model Response 프롬프트 목록:")
    model_response_list = manager.get_prompt_list("model_response")
    print(f"Available: {model_response_list}")
    
    # 기존 호환성을 위한 결합된 세트 확인
    print("\n4. 결합된 프롬프트 세트:")
    combined_sets = manager.get_combined_prompt_sets()
    for prompt_set in combined_sets:
        print(f"- {prompt_set.get('name', 'unnamed')}")
    
    print("\n=== 테스트 완료 ===")