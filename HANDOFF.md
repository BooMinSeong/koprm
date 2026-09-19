# 인수인계 (2026-09-20, 서버 이전용)

새 세션은 이 문서와 `Plan.md`(v9.2, 특히 §13.1의 결정 기록)를 읽고 §5의 순서대로 진행한다. 코드 작성은 Opus 에이전트에 맡기고 메인 세션은 판단·검토·실행 결정을 맡는다(사용자 지침).

## 사용자 지침 (반드시 지킬 것)

- 번역기는 하나만 쓴다. 폴백 모델, 재시도 패스, 임베딩 유사도 검사(LaBSE) 같은 이중 장치는 넣지 않는다. 문제의 원인은 파싱·로직에서 고친다.
- 중요한 변경이나 논의 사항이 있으면 멈추고 사용자에게 알린다. 그 외에는 자율 진행.
- 커밋은 https://github.com/BooMinSeong/koprm.git 의 `main`에 한다. 데이터(`data/`)는 커밋하지 않는다.

## 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 번역기 | gemma-3-12b-it, 지시 프롬프트 + 자리표시자 마스킹 | 300문제 복원율 96.3%; TranslateGemma 89.0%(지시 불가), Qwen2.5-7B 94.8% |
| `$...$` 판정 | 내용 기반(산문 3단어 연속, 한글, 줄바꿈, `$5 … $8` 통화 짝이면 수식 아님) | GSM8K 문제의 18%가 통화 기호로 오인됨 |
| 교사 | Qwen2.5-Math-PRM-72B, vLLM `runner="pooling"`, `PoolerConfig(use_activation=False)`, 헤드 fp32, fp8 가중치 양자화(Ampere), TP=2 | 7B로 HF 참조와 대조: 스텝 수 일치, \|Δz\| ≤ 0.2 |
| 학생 | EXAONE-4.0-1.2B + `Linear-ReLU-Linear(2)` 헤드, 구분 토큰 `[unused0]`(id 62) | 1일차 점검 통과, 폴백 불필요 |
| 평가 | komath 저장 64샘플(`ENSEONG/ko-ko-math-500-test-<gen>-bon`)을 학생으로 재채점 | 재구현 집계가 komath 결과를 1.4%p 이내로 재현 |
| 감사 지표 | y=0 풀이의 마지막 이전 스텝 중 마스크되지 않은 라벨의 정밀도 | 마지막 스텝·y=1 풀이는 규칙상 자동 일치라 제외 |

## 코드 지도

```
koprm/data/sources.py, splits.py      MATH·GSM8K·KO MATH500 로드, dev/train 풀 분할 (§4.1)
koprm/data/prm800k.py, prm800k_sets.py PRM800K phase2 파싱, 감사 세트(오답500+정답500)·A 풀 (§4.3, §4.4)
koprm/gen/generate.py, outcome.py     vLLM 한국어 생성(komath 프롬프트), math_verify 병렬 y 채점
koprm/translate/mask.py, translate.py 마스킹/복원, 단일 번역기, 재개 가능 jsonl
koprm/teacher/score.py                교사 로그 오즈 (§2.2)
koprm/label/kernel.py, build.py, audit.py  기준 분포·커널 (§2.3–2.4), 라벨 조립, 감사 보고
koprm/select.py, trainsets.py         §4.2 선택, §4.5 학습 세트(3k⊂6k⊂12k, A/B/A+B)
koprm/train/{model,data,loss,train}.py 학생 모델·손실(§6.1)·학습 루프
koprm/eval/{scorer,bon}.py            학생 채점기, BoN 재채점 평가(naive/weighted/maj, 부트스트랩)
scripts/check_teacher.py, check_student.py  1일차 점검
```

테스트: `.venv/bin/python -m pytest tests -q` (84개 이상 통과해야 함).

## 새 서버에서 먼저 할 일

1. `git clone` 후 `uv sync --extra dev`. HF 토큰은 사용자 계정(ENSEONG). `HF_HUB_ENABLE_HF_TRANSFER=1`.
2. 모델 다운로드를 즉시 시작: `Qwen/Qwen2.5-Math-PRM-72B`(139GB, 가장 오래 걸림), `google/gemma-3-12b-it`, `LGAI-EXAONE/EXAONE-4.0-1.2B`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-Math-PRM-7B`(점검용). 데이터셋: `openai/gsm8k`, `HuggingFaceH4/MATH-500`, `ENSEONG/ko-math-500-test`, `tasksource/PRM800K`, `ENSEONG/ko-ko-math-500-test-{EXAONE-4.0-1.2B,Qwen2.5-3B-Instruct,Qwen2.5-1.5B-Instruct}-bon`. MATH 원본은 `EleutherAI/hendrycks_math`로 폴백된다(`KOPRM_MATH_DIR`로 로컬 사본 지정 가능).
3. `python -m koprm.data.splits` → `python -m koprm.data.prm800k_sets` (수 분).
4. 문제 번역: `data/trans/problems_in.jsonl`(dev+train_pool, id=problem_id)을 만들고
   `python -m koprm.translate.translate --in data/trans/problems_in.jsonl --out data/trans/problems_ko.jsonl --field problem_en --out-field problem_ko --src en --tgt ko --primary-model google/gemma-3-12b-it --primary-backend instruct` (1 GPU 약 30분).
5. 이후 §5 순서: 생성(dev 500×2생성기×16, train 풀 14,473×2×2; `--shard/--num-shards`로 GPU 분할) → y 채점 → 감사 세트 왕복(영→한→영)과 72B 채점 → 커널 정밀도 확인(§9 관문: 90%) → 선택 → B 라벨링 → A 번역 → 학습 세트 → 학습 11회 → dev 선택 → MATH500 재채점.

## 알려진 주의점

- transformers 4.57 + `HF_HUB_OFFLINE=1`에서 `AutoTokenizer.from_pretrained(repo_id)`가 실패할 수 있다. `koprm/train/model.py`의 `load_tokenizer()`가 우회한다.
- Qwen2.5-Math-PRM의 원격 코드는 transformers 4.57에서 `use_cache=False`가 필요하다(점검 스크립트에 반영됨).
- 번역된 영어 스텝의 6~8%에 `\text{인치}`처럼 마스킹된 수식 안의 한글이 남는다. 손대지 않고 한계로 둔다.
- EXAONE 생성기는 프롬프트의 "[간결한 설명]"을 그대로 베끼는 버릇이 있다. 무해하다.
- 파일럿 정답률(EXAONE, KO MATH500 50문제): 56%. math_verify 오탐 0건 확인.
