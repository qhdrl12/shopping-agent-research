# Railway 배포 가이드

## 1. Railway 계정 설정
1. [Railway](https://railway.app) 가입
2. GitHub 계정 연결
3. 신용카드 등록 (무료 크레딧 활성화용)

## 2. 프로젝트 배포
1. Railway 대시보드에서 "New Project" 클릭
2. "Deploy from GitHub repo" 선택
3. `shopping-agent-research` 저장소 선택
4. 자동 배포 시작

## 3. 필수 환경변수 설정
Railway 대시보드 → Variables 탭에서 다음 변수들을 설정:

```
OPENAI_API_KEY=sk-proj-your-key-here
FIRECRAWL_API_KEY=fc-your-key-here
TAVILY_API_KEY=tvly-your-key-here
LANGSMITH_TRACING=false
```

## 4. 배포 상태 확인
- Railway 대시보드에서 배포 로그 확인
- 성공 시 제공되는 URL로 접속
- 실패 시 로그에서 에러 메시지 확인

## 5. 비용 관리
- 무료 플랜: 월 $5 크레딧
- 사용량 모니터링: Railway 대시보드 → Usage 탭
- 필요시 앱 일시 정지: Deploy → Settings → Sleep

## 6. 문제 해결
- **빌드 실패**: requirements.txt 의존성 확인
- **런타임 에러**: 환경변수 설정 확인
- **메모리 초과**: Railway Pro 플랜 고려 ($20/월)

## 7. URL 확인
배포 완료 후 Railway에서 제공하는 URL:
`https://your-app-name.up.railway.app`