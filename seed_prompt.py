import os
import sys
from dotenv import load_dotenv

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.local_prompt_manager import LocalPromptManager

load_dotenv()

def seed_default_prompt():
    """로컬 파일에 기본 프롬프트를 생성합니다."""
    try:
        manager = LocalPromptManager()
        
        # 1. 기존 'default' 프롬프트가 있는지 확인
        existing_prompt = manager.get_prompt('default')
        if existing_prompt:
            print("✅ 'default' 프롬프트가 이미 존재합니다. 업데이트를 시도합니다.")
            
            # Load prompts from default.json
            import json
            with open('src/config/prompts/default.json', 'r', encoding='utf-8') as f:
                default_prompts = json.load(f)
            
            query_analysis_prompt = default_prompts['analysis_prompt_template']
            model_response_prompt = default_prompts['system_prompt_template']
            
            manager.update_prompt(
                prompt_id=existing_prompt['id'],
                name='default',
                query_analysis_prompt=query_analysis_prompt,
                model_response_prompt=model_response_prompt
            )
            print("🔄 'default' 프롬프트가 성공적으로 업데이트되었습니다.")

        else:
            print("ℹ️ 'default' 프롬프트가 존재하지 않아 새로 생성합니다.")
            # Load prompts from default.json
            import json
            with open('src/config/prompts/default.json', 'r', encoding='utf-8') as f:
                default_prompts = json.load(f)
            
            query_analysis_prompt = default_prompts['analysis_prompt_template']
            model_response_prompt = default_prompts['system_prompt_template']

            manager.create_prompt(
                name='default',
                query_analysis_prompt=query_analysis_prompt,
                model_response_prompt=model_response_prompt
            )
            print("✨ 'default' 프롬프트가 성공적으로 생성되었습니다.")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    seed_default_prompt()
