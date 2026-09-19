# koprm — 한국어 PRM 라벨 투영

영어 PRM(Qwen2.5-Math-PRM-72B)의 판정을 번역 다리로 빌려 한국어 풀이에 하드 라벨을 붙이고, 그 라벨로 한국어 PRM을 학습한다. 설계는 [Plan.md](Plan.md)를 따른다.

## 환경

```bash
uv sync --extra dev          # vllm 0.15, transformers 4.57, math-verify 0.9
.venv/bin/python -m pytest tests -q
```

공유 서버라 GPU는 `CUDA_VISIBLE_DEVICES`로 명시해서 쓴다. HF 캐시는 `HF_HOME`(공유)이다.

## 파이프라인 (Plan §5)

| 단계 | 모듈 | 산출물 (`data/`) |
|---|---|---|
| 1 분할 | `koprm.data.splits` | `splits/{dev,train_pool,math500}.jsonl` |
| 2 문제 번역 영→한 | `koprm.prep problems-in` → `koprm.translate.translate` | `trans/problems_ko.jsonl` |
| 3 생성 + y 채점 | `koprm.prep problems-ko` → `koprm.gen.generate`, `koprm.gen.outcome` | `gen/<gen>.scored.jsonl` |
| 4 감사 세트 | `koprm.data.prm800k` + 번역 왕복 + 교사 + 커널 | `labels/audit.json` |
| 5 선택 | `koprm.select` | `gen/selected.jsonl` |
| 6 B 라벨링 | 스텝 번역 한→영 → `koprm.prep teacher-in` → `koprm.teacher.score` → `koprm.label.kernel` | `labels/B.jsonl` |
| 7 A 구성 | `koprm.data.prm800k` + 번역 영→한 | `labels/A.jsonl` |
| 8 학습 세트 | `koprm.trainsets` | `trainsets/{A,B,AB}_{3k,6k,12k}.jsonl` |
| 9 학습 | `koprm.train.train` | `ckpt/` |
| 10 dev 선택 / 11 MATH500 | `koprm.eval.bon` (저장된 64샘플 재채점) | `eval/` |

단계 사이의 입력 jsonl은 `koprm.prep`의 하위 명령(`problems-in`, `problems-ko`, `prm800k-problems-in`, `teacher-in`)으로 앞 단계 산출물에서 만든다. 모든 단계는 jsonl을 `id` 기준으로 재개할 수 있고, 생성·번역은 `--shard i --num-shards k`로 GPU별 프로세스를 나눠 돌린다.

## 참고 구현

- `../komath`: KO MATH500 테스트타임 컴퓨트 하네스. 생성 프롬프트, `\n\n` 스텝 분절, math_verify 채점 방식, BoN 집계(naive/weighted/maj)를 그대로 따른다.
