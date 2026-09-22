# 인수인계 — §5 전 단계 (2026-09-20), 확장 실험·분포 이동 (2026-09-21), 큰 학생 (2026-09-22)

분할부터 KO MATH500 재채점까지 §5의 전 단계를 실행했고(2026-09-20), 이어서 집계 방식·데이터 확장(4.8만)·3B 백본·소프트+y의 확장 실험과 분포 이동 평가(영어 MATH500, 한국어 AIME, 한국어 ProcessBench)를 돌렸다(2026-09-21). 이어서 7~8B 학생과 강한 생성기를 돌렸다(2026-09-22). 결과와 판정은 `Plan.md` §15(확장 실험 §15.7, 분포 이동 §15.8, 큰 학생 §15.9)에, 모든 표는 `data/reports/results.md`에 있다. 새 세션은 이 문서와 `Plan.md`(§15 결과, §13.0·§13.1 결정 기록)를 읽고 "다음 세션이 할 일"부터 이어 간다. 코드 작성은 Opus 에이전트에 맡기고 메인 세션은 판단·검토·실행 결정을 맡는다(사용자 지침).

## 핵심 결과 (자세한 내용은 Plan §15)

세 가지 사전 등록 비교(§1) 중 "같은 크기에서 B ≥ A"는 성립하고(차이가 0을 포함), "A+B가 둘 이상"은 3k·6k의 naive@16에서만 유의하며, 학습 곡선은 6천에서 1.2만 사이가 평평해 §1의 두 번째 증거는 성립하지 않는다. 이번 실행의 주 발견은 절제다. KO MATH500·EXAONE naive@64에서 현행(영어 PRM 직접) 0.664, 커널 하드 라벨 B_12k 0.520, 교사 소프트 타깃 0.692, 결과 항만 쓴 학생 0.740이다. 결과 항만 쓴 학생이 모든 지표에서 현행을 넘고 n=64에서 꺾이지 않는 유일한 학생이며, 같은 순서가 Qwen2.5-3B와 홀드아웃 Qwen2.5-1.5B에서도 나온다. 번역 다리를 건넌 교사(72B, min, n=16)는 naive@16 0.762로 현행 0.670보다 높아, 다리가 아니라 학생과 라벨 형식이 병목이다.

확장 실험(§15.7)에서 그 병목을 라벨 형식과 데이터 양으로 풀었다. B 풀을 4.8만 풀이로 늘리자 KO MATH500·EXAONE에서 교사 소프트가 naive@64 0.744, 결과 항만이 weighted@64 0.780으로 현행(0.664 / 0.718)을 크게 넘는다. 두 형식 모두 24k→48k에서 통계적으로 유의하게 올라 아직 포화하지 않았다. 하드 커널 라벨은 데이터를 네 배로 늘려도, 집계를 바꿔도 회복되지 않는다. Qwen2.5-3B 백본은 같은 초매개변수에서 1.2B보다 모든 라벨 형식에서 나빴고, 소프트+y는 소프트를 넘지 못했다.

큰 학생(§15.9)이 이번 라운드의 답이다. Qwen3-8B 백본에 같은 B_24k 소프트 라벨로 4장 FSDP 전체 미세조정을 하면 KO MATH500·EXAONE naive@64가 0.796으로, 현행 영어 PRM(0.664)은 물론 4.8만 풀이로 학습한 1.2B(0.744)도 넘는다. 같은 데이터의 1.2B 대비 +0.082 [+0.052, +0.114]로 유의하고, n=16에서는 번역 다리를 건넌 72B 교사(0.762)보다도 높다(0.772). 영어 PRM에서 출발한 7B는 그보다 낮고(0.748) dev가 오염되어 있어 비교에 쓸 수 없다. 스텝 단위 판정도 같이 올라간다. 한국어 ProcessBench F1이 8B 학생 0.638로 영어 PRM 7B를 직접 쓴 0.579를 넘었고, 72B 직접(0.746)만 아직 위다.

분포 이동(§15.8)에서는 한국어로 학습한 학생을 고치지 않고 옮겼다. 영어 MATH500에서 B_48k_soft가 naive@64 0.826으로 영어 PRM 7B(0.830)에 0.4%p까지 붙고 weighted@64에서는 앞선다. 한국어 ProcessBench F1은 72B 0.746, 7B 0.579, 소프트 학생 0.511로 스텝 단위 판정이 아직 뒤진다(임계 0.5는 보정 전이다). 한국어 AIME는 생성기가 너무 약해(EXAONE pass@64 0.411) 채점기를 가르지 못했다.

## 산출물 위치

```
data/splits/      dev, train_pool, math500, prm800k_audit, prm800k_A_pool
data/trans/       문제·스텝 번역 (problems_ko, selected.steps_en, prm800k_A.steps_ko,
                  audit.steps_ko, audit.steps_en_rt, math500_exaone16.steps_en)
data/gen/         생성 결과와 y 채점, selected.jsonl
                  확장: train_*_n6*(생성기당 6표본 추가), selected_big*.jsonl
data/teacher/     72B 로그 오즈 (B, 감사 rt/direct, MATH500 참조선)
data/labels/      A.jsonl, B.jsonl, B.refs.json, audit_rows.jsonl, B_big.jsonl(4.8만)
data/trainsets/   {A,B,AB}_{3k,6k,12k}.jsonl, big/B_{12k,24k,48k}.jsonl
data/ckpt/<run>/epoch{1,2,3}   원래 11회 + B_12k_soft_y + big_B_{12,24,48}k{,_soft,_soft_y,
                  _outcome} + q3b_*(3B 백본)
data/eval/dev/, math500/, selection.json, cache/
data/eval/dev_big/, math500_big/   4.8만 계열
data/eval/dev3b/, math500_3b/      3B 백본
data/eval/agg/    원래 체크포인트의 네 집계(last/min/prod/mean)
data/eval/dev7b/, math500_7b/      7B·8B 학생
data/ckpt/prm7b_B_24k_soft, qwen3-8b_B_24k_soft   7B·8B 학생 체크포인트
data/models/EXAONE-3.5-7.8B-Instruct-patched      가중치 심링크 + 패치 시도(실패)
data/shift/       분포 이동: math500_en_problems.jsonl, gen_math500_en_*,
                  math500_en_*.bon.jsonl, aime_problems_ko.jsonl, gen_aime_ko_*,
                  aime_ko_*.bon.jsonl, pb_rows.jsonl, en/, aime/(강한 생성기 *qwen3* 포함), pb/
data/trans/       (분포 이동분) aime_in, aime_ko, pb_in, pb_problems_ko, pb_steps_ko
data/teacher/     (분포 이동분) pb_* (7B·72B의 ProcessBench 로그 오즈)
data/reports/results.md, results.json, audit.json, student_audit_<run>.json
logs/*.sh         단계별 드라이버(gen_all, translate_*, teacher_*, labels, train_all,
                  post_eval, final_eval, reference_line)와 확장용 scale_stage*.sh,
                  train_3b.sh, train_softy.sh, 분포 이동용 shift_*.sh·shift_glue.py,
                  큰 학생용 train_7b*.sh와 aime_*strong*.sh, 그리고 그 로그
```

`data/reports/`만 커밋한다. 나머지 `data/`와 `logs/`는 커밋하지 않는다.

## 사용자 지침 (반드시 지킬 것)

- 번역기는 하나만 쓴다. 폴백 모델, 재시도 패스, 임베딩 유사도 검사(LaBSE) 같은 이중 장치는 넣지 않는다. 문제의 원인은 파싱·로직에서 고친다.
- 중요한 변경이나 논의 사항이 있으면 멈추고 사용자에게 알린다. 그 외에는 자율 진행.
- 커밋은 https://github.com/BooMinSeong/koprm.git 의 `main`에 한다. 데이터(`data/`)는 커밋하지 않는다.

## 환경

vLLM 0.29.0 + transformers 5.17 + torch 2.13의 단일 venv(`.venv`)를 쓴다. gemma-4-12B-it는 vLLM 0.15의 레지스트리에 없고(`Gemma4UnifiedForConditionalGeneration`), vLLM 0.29는 transformers 5.10.4 이상을 요구한다. 이 서버에서는 flashinfer 샘플러의 JIT 빌드가 ninja·nvcc를 요구해 실패하므로 `VLLM_USE_FLASHINFER_SAMPLER=0`으로 돌린다. `koprm/paths.py`가 임포트 시점에 `os.environ.setdefault`로 설정하므로, vLLM을 쓰는 모듈(생성·번역·교사)은 그대로 실행하면 된다. GPU는 A6000 48GB 8장, HF 캐시는 공유 경로 `/srv/lilab/hf_cache`다.

## 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 번역기 | gemma-4-12B-it, 지시 프롬프트 + 자리표시자 마스킹 | 같은 300문제(MATH 150 + GSM8K 150) 복원율: gemma-4-12B-it 99.3%(298/300) → 프롬프트·파싱 수정 후 99.7%(299/300), gemma-3-12b-it 93.7%(281/300) → 97.0%(291/300). gemma-3 성공 문제는 전부 gemma-4도 성공 |
| `$...$` 판정 | 내용 기반(산문 3단어 연속, 한글, 줄바꿈, `$5 … $8` 통화 짝이면 수식 아님) | GSM8K 문제의 18%가 통화 기호로 오인됨 |
| 교사 | Qwen2.5-Math-PRM-72B, vLLM `runner="pooling"`, `PoolerConfig(use_activation=False)`, 헤드 fp32, bf16 TP=4(8장 환경). 2장만 쓸 때는 fp8 가중치 양자화(Ampere) + TP=2 | 7B로 HF 참조와 대조: 스텝 수 일치, \|Δz\| ≤ 0.2 |
| 학생 | EXAONE-4.0-1.2B + `Linear-ReLU-Linear(2)` 헤드, 구분 토큰 `[unused0]`(id 62) | 1일차 점검 통과, 폴백 불필요 |
| 평가 | komath 저장 64샘플(`ENSEONG/ko-ko-math-500-test-<gen>-bon`)을 학생으로 재채점 | 재구현 집계가 komath 결과를 1.4%p 이내로 재현 |
| 감사 지표 | y=0 풀이의 마지막 이전 스텝 중 마스크되지 않은 라벨의 정밀도 | 마지막 스텝·y=1 풀이는 규칙상 자동 일치라 제외 |

## 코드 지도

```
koprm/data/sources.py, splits.py      MATH·GSM8K·KO MATH500 로드, dev/train 풀 분할 (§4.1)
koprm/data/prm800k.py, prm800k_sets.py PRM800K phase2 파싱, 감사 세트(오답500+정답500)·A 풀 (§4.3, §4.4)
koprm/gen/generate.py, outcome.py     vLLM 한국어 생성(komath 프롬프트), math_verify 병렬 y 채점
                                      `--sample-offset N`으로 2차 생성의 id 충돌을 막는다
                                      `--problem-field`/`--system-prompt {ko,en}`(영어 생성)
                                      `--chat-kwargs '{"enable_thinking": false}'`(추론 모델)
                                      생성기 별칭에 qwen3-8b, exaone-32b 추가
koprm/translate/mask.py, translate.py 마스킹/복원, 단일 번역기, 재개 가능 jsonl
koprm/teacher/score.py                교사 로그 오즈 (§2.2)
                                      `--problem-field`/`--steps-field`(한국어 스텝 직접 채점)
koprm/label/kernel.py, build.py, audit.py  기준 분포·커널 (§2.3–2.4), 라벨 조립, 감사 보고
koprm/select.py, trainsets.py         §4.2 선택(`--per-class K`), §4.5 학습 세트
                                      (`--sizes 12k,24k,48k --arms B`로 새 크기만 만든다)
koprm/train/{model,data,loss,train}.py 학생 모델·손실(§6.1)·학습 루프
                                      라벨 형식: `--soft` / `--soft-y` / `--outcome-only`(배타)
                                      `--fsdp`(torchrun, FSDP2 전체 미세조정),
                                      `--init-from-prm`(영어 PRM 백본 + 헤드, §12.2)
koprm/eval/{scorer,bon}.py            학생 채점기, BoN 재채점 평가(naive/weighted/maj, 부트스트랩)
                                      `--agg all`(last/min/prod/mean), `--save-scores`/`--scores-from`
                                      `--system-prompt {ko,en}`(학생 템플릿)
koprm/prep.py                         단계 사이의 입력 jsonl 조립(problems-in/ko, teacher-in 등)
koprm/report.py                       §7 보고 항목 조립(results.md/json), 교사 참조선, 학생 첫 오류 감사
koprm/shift.py                        분포 이동: math500-en, bon-jsonl, processbench-rows,
                                      first-error(학생 또는 교사, ProcessBench err/corr/F1)
scripts/check_teacher.py, check_student.py  1일차 점검
```

테스트: `.venv/bin/python -m pytest tests -q` (161개 통과해야 함).

## 재현 순서 (새 서버에서 처음부터 돌릴 때)

1. `git clone` 후 `uv sync --extra dev`(vLLM 0.29.0, transformers 5.17). HF 토큰은 사용자 계정(ENSEONG).
2. 모델 다운로드를 즉시 시작: `Qwen/Qwen2.5-Math-PRM-72B`(139GB, 가장 오래 걸림), `google/gemma-4-12B-it`(게이트 아님), `LGAI-EXAONE/EXAONE-4.0-1.2B`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-Math-PRM-7B`(점검용). 데이터셋: `openai/gsm8k`, `HuggingFaceH4/MATH-500`, `ENSEONG/ko-math-500-test`, `tasksource/PRM800K`, `ENSEONG/ko-ko-math-500-test-{EXAONE-4.0-1.2B,Qwen2.5-3B-Instruct,Qwen2.5-1.5B-Instruct}-bon`. MATH 원본은 `EleutherAI/hendrycks_math`로 폴백된다(`KOPRM_MATH_DIR`로 로컬 사본 지정 가능).
3. `python -m koprm.data.splits` → `python -m koprm.data.prm800k_sets` (수 분).
4. 문제 번역: `data/trans/problems_in.jsonl`(dev+train_pool, id=problem_id)을 만들고
   `python -m koprm.translate.translate --in data/trans/problems_in.jsonl --out data/trans/problems_ko.jsonl --field problem_en --out-field problem_ko --src en --tgt ko --primary-model google/gemma-4-12B-it --primary-backend instruct` (1 GPU 약 30분).
5. 이후 §5 순서: 생성(dev 500×2생성기×16, train 풀 14,473×2×2; `--shard/--num-shards`로 GPU 분할) → y 채점 → 감사 세트 왕복(영→한→영)과 72B 채점 → 커널 정밀도 확인(§9 관문: 90%) → 선택 → B 라벨링 → A 번역 → 학습 세트 → 학습 11회 → dev 선택 → MATH500 재채점.

## 다음 세션이 할 일

기본 구성은 Qwen3-8B 백본에 교사 소프트 라벨, 풀이 2.4만 이상이다(§15.9d). 하드 커널 라벨과 1.2B 학생은 대체되었다. 임계 보정과 영어 템플릿 변형은 사용자 판단으로 제외한다(스텝 판정은 학습으로 얻어야 하고, 템플릿은 본질이 아니다).

1. Qwen3-8B를 4.8만 풀이로 올린다(GPU 4장에서 약 24시간). 8B·2.4만이 이미 1.2B·4.8만을 넘었으므로 둘을 합치는 것이 가장 확실한 다음 수다.
2. 8B 학생으로 ProcessBench를 다시 잰다. 스텝 단위 판정이 크기와 함께 올라가는지가 남은 질문이다.
3. 9.6만 지점을 찍는다. 남은 2.8만 풀에 표본을 더해 `--per-class`를 올려 뽑는다. 이제는 1.2B가 아니라 8B로 찍는다.
4. 첫 오류 감사를 한국어 문제 텍스트로 다시 잰다(지금 수치는 영어 문제 조건이라 학생에게 불리하다).
5. AIME는 보류한다. Qwen3-4B·Qwen3-8B로 표본당 정답률은 올랐지만 pass@64가 0.39~0.41에 머물러(56문제 중 같은 23문제) 채점기를 가르지 못한다(§15.9c).

무엇을 할지는 사용자가 정한다. 그리고 이 서버에는 자격 증명이 없어 GitHub push가 밀려 있다. 로컬에 30개가 넘는 커밋이 쌓여 있으므로 다음 세션에서 `main`으로 push한다.

## 알려진 주의점

- `HF_HUB_OFFLINE=1`에서 `AutoTokenizer.from_pretrained(repo_id)`가 실패할 수 있다. `koprm/train/model.py`의 `load_tokenizer()`가 로컬 스냅샷 경로로 우회한다.
- `apply_chat_template(tokenize=True)`는 `BatchEncoding`(dict이 아니라 UserDict)을 돌려준다. `encode_example()`은 `Mapping`으로 판정한다.
- Qwen2.5-Math-PRM의 HF 원격 코드는 transformers 5에서 돌지 않을 수 있다. `scripts/check_teacher.py`의 `--save-json`/`--ref-json`으로 저장한 로그 오즈와 대조한다(0.15 대비 16/16 스텝 일치, 최대 |Δz| = 0.077).
- 번역된 영어 스텝의 6~8%에 `\text{인치}`처럼 마스킹된 수식 안의 한글이 남는다. 손대지 않고 한계로 둔다.
- EXAONE 생성기는 프롬프트의 "[간결한 설명]"을 그대로 베끼는 버릇이 있다. 무해하다.
- 파일럿 정답률(EXAONE, KO MATH500 50문제): 56%. math_verify 오탐 0건 확인.
- FSDP는 이 서버에서 GPU 4장이 8장보다 빠르다(스텝당 39~43초 대 60초). all-gather가 PCIe에 묶인다.
- EXAONE-3.5-7.8B의 원격 코드는 transformers 5.17과 맞지 않는다(`create_causal_mask(input_embeds=...)`). 로드·저장·재로드는 되지만 순전파가 안 된다. EXAONE-4.0에는 중간 크기 모델이 없다.
- `--init-from-prm`(Qwen2.5-Math-PRM-7B)로 학습한 학생은 dev가 오염되어 있다. 그 PRM이 MATH train으로 학습되었기 때문이다. 에폭을 dev로 고르지 말고 미리 선언하고, MATH·GSM8K train에서 뽑은 어떤 비교에도 기준선으로 쓰지 않는다.
- 이 서버는 다른 사용자와 공유한다. 작업 전에 `nvidia-smi`로 빈 GPU를 확인하고 `CUDA_VISIBLE_DEVICES`를 명시한다.
- vLLM 프로세스를 `pkill`로 끊으면 워커가 고아로 남아 GPU 메모리를 잡고 있다. 다음 작업 전에 `nvidia-smi`로 확인하고 남은 워커를 직접 정리한다.
- `koprm/eval/bon.py`는 `load_hf_rows`에서 `HF_HUB_OFFLINE=1`을 setdefault한다. 저장된 BoN 데이터셋을 새로 받아야 하면 `HF_HUB_OFFLINE=0`으로 실행한다(`koprm.report teacher-ref`는 스스로 0으로 설정한다).
- 저장된 64샘플 데이터셋의 config 이름은 `ENSEONG_ko-math-500-test--T-0.8--top_p-1.0--n-64--seed-{0,42,64}--agg_strategy-last`다. 이번 실행은 seed 0을 썼다.
