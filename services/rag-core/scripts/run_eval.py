"""Golden-set evaluation harness (Stage 8).

Runs the golden Q/A set (eval/golden_set.yaml) through either pipeline
(improved by default, naive with --pipeline naive). Grounded questions
(check: ragas) are scored with Ragas (faithfulness, answer_relevancy,
context_precision, context_recall); no-answer/adversarial/conflicting-source/
supersedes questions (check: refusal|keyword) are checked deterministically -
a single reference answer doesn't fit those categories, same reasoning
Stage 6 used for its live tests.

Requires the real corpus ingested (corpus/files/, not just corpus/sample/) -
`make eval` handles ingestion first. Explicitly triggered only, per
DIRECTION.md's evaluation standing rule - never run on every change, never in
CI, never in a background job.

Usage (from services/rag-core/, inside the container or venv):
    uv run --frozen python -m scripts.run_eval
    uv run --frozen python -m scripts.run_eval --limit 8
    uv run --frozen python -m scripts.run_eval --pipeline naive
    uv run --frozen python -m scripts.run_eval --ids f01,f02,n01
    uv run --frozen python -m scripts.run_eval --category factual
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # disable ragas' anonymous usage telemetry

import yaml

from app.config import Settings, get_settings
from app.generation.claude import generate_answer, make_anthropic_client
from app.naive.generation import naive_generate
from app.naive.search import naive_search
from app.providers import get_embedding_provider
from app.providers.base import EmbeddingProvider
from app.providers.rerank import get_rerank_provider
from app.providers.rerank.base import RerankProvider
from app.retrieval.search import search

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.yaml"
RESULTS_DIR = EVAL_DIR / "results"

RAGAS_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class PipelineOutput:
    answer: str
    contexts: list[str]
    stop_reason: str | None  # naive has no refusal gating - always None


@dataclass
class QuestionResult:
    id: str
    category: str
    check: str
    question: str
    answer: str
    passed: bool | None = None
    detail: str = ""
    ragas_scores: dict[str, float | None] | None = None
    error: bool = False


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["questions"]


def run_improved(
    question: str,
    *,
    settings: Settings,
    embedder: EmbeddingProvider,
    reranker: RerankProvider,
    anthropic_client,
) -> PipelineOutput:
    chunks = search(question, settings=settings, embedder=embedder, reranker=reranker)
    answer = generate_answer(question, chunks, settings=settings, client=anthropic_client)
    return PipelineOutput(
        answer=answer.text,
        contexts=[c.chunk_text for c in chunks],
        stop_reason=answer.stop_reason,
    )


def run_naive(
    question: str, *, settings: Settings, embedder: EmbeddingProvider, anthropic_client
) -> PipelineOutput:
    chunks = naive_search(question, settings=settings, embedder=embedder)
    answer = naive_generate(question, chunks, settings=settings, client=anthropic_client)
    return PipelineOutput(answer=answer, contexts=chunks, stop_reason=None)


def check_keyword(answer: str, item: dict) -> tuple[bool, str]:
    lowered = answer.lower()
    expected = item.get("expected_phrases", [])
    forbidden = item.get("forbidden_phrases", [])
    missing = [p for p in expected if p.lower() not in lowered]
    present = [p for p in forbidden if p.lower() in lowered]
    if missing or present:
        return False, f"missing={missing} forbidden_found={present}"
    return True, "ok"


def check_refusal(stop_reason: str | None, item: dict) -> tuple[bool, str]:
    expected = item.get("expected_stop_reasons", [])
    return stop_reason in expected, f"stop_reason={stop_reason!r} expected one of {expected}"


def build_ragas_metrics(settings: Settings) -> dict:
    import anthropic
    from ragas.embeddings import LiteLLMEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    # Ragas' modern metrics call the client's async methods (.ascore() ->
    # .agenerate()) - the synchronous anthropic.Anthropic client used
    # elsewhere in this script raises "Cannot use agenerate() with a
    # synchronous client", found live, not documented anywhere obvious.
    async_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    llm = llm_factory(settings.generation_model, provider="anthropic", client=async_client)
    # claude-sonnet-5 removed `temperature`/`top_p` from the Messages API
    # entirely (confirmed via the claude-api skill's model-capability table -
    # not guessed), but ragas' InstructorModelArgs defaults temperature=0.01
    # unconditionally and llm_factory has no public option to omit it. Left
    # in place, every ragas call raises "AsyncMessages.create() got an
    # unexpected keyword argument 'temperature'" - a client-side TypeError,
    # found live, before any request reaches the API. Strip both directly;
    # `max_tokens` (still a real, supported param) is left untouched.
    llm.model_args.pop("temperature", None)
    llm.model_args.pop("top_p", None)
    # litellm's "voyage/<model>" provider prefix reuses VOYAGE_API_KEY - no new
    # embedding provider added just for this one Ragas metric.
    embeddings = LiteLLMEmbeddings(
        model=f"voyage/{settings.embedding_model}", api_key=settings.voyage_api_key
    )
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def score_ragas(
    metrics: dict, question: str, answer: str, contexts: list[str], ground_truth: str
) -> dict[str, float | None]:
    if not contexts:
        # Nothing retrieved - faithfulness/precision/recall are undefined
        # against an empty context set, not zero.
        return dict.fromkeys(RAGAS_METRIC_NAMES)

    coros = {
        "faithfulness": metrics["faithfulness"].ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        ),
        "answer_relevancy": metrics["answer_relevancy"].ascore(user_input=question, response=answer),
        "context_precision": metrics["context_precision"].ascore(
            user_input=question, reference=ground_truth, retrieved_contexts=contexts
        ),
        "context_recall": metrics["context_recall"].ascore(
            user_input=question, retrieved_contexts=contexts, reference=ground_truth
        ),
    }
    scores: dict[str, float | None] = {}
    for name, coro in coros.items():
        try:
            result = await coro
            scores[name] = float(result.value)
        except Exception as exc:  # noqa: BLE001 - one metric failing shouldn't sink the whole question
            print(f"  ! ragas metric {name!r} failed: {exc}", file=sys.stderr)
            scores[name] = None
    return scores


def _detail_for_ragas(scores: dict[str, float | None]) -> str:
    return ", ".join(f"{k}={v:.2f}" if v is not None else f"{k}=n/a" for k, v in scores.items())


async def run_all(args: argparse.Namespace) -> list[QuestionResult]:
    settings = get_settings()
    questions = load_golden_set()

    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("No questions matched the given filters.", file=sys.stderr)
        return []

    anthropic_client = make_anthropic_client(settings.anthropic_api_key)
    embedder = get_embedding_provider(settings)
    reranker = get_rerank_provider(settings) if args.pipeline == "improved" else None
    ragas_metrics = build_ragas_metrics(settings)

    results: list[QuestionResult] = []
    for item in questions:
        print(f"[{item['id']}] ({item['category']}/{item['check']}) {item['question']}")

        # One question's provider error (e.g. a rate limit - a real,
        # repeatedly observed condition on this project's Voyage account)
        # must not abort the rest of the golden set, same per-item isolation
        # principle as the ingestion pipeline.
        try:
            if args.pipeline == "improved":
                out = run_improved(
                    item["question"],
                    settings=settings,
                    embedder=embedder,
                    reranker=reranker,
                    anthropic_client=anthropic_client,
                )
            else:
                out = run_naive(
                    item["question"],
                    settings=settings,
                    embedder=embedder,
                    anthropic_client=anthropic_client,
                )
        except Exception as exc:  # noqa: BLE001 - isolate this question, keep the run going
            print(f"  ! pipeline error, skipping: {exc}", file=sys.stderr)
            results.append(
                QuestionResult(
                    id=item["id"],
                    category=item["category"],
                    check=item["check"],
                    question=item["question"],
                    answer="",
                    detail=f"pipeline error: {exc}",
                    error=True,
                )
            )
            continue

        result = QuestionResult(
            id=item["id"],
            category=item["category"],
            check=item["check"],
            question=item["question"],
            answer=out.answer,
        )

        if item["check"] == "keyword":
            result.passed, result.detail = check_keyword(out.answer, item)
        elif item["check"] == "refusal":
            result.passed, result.detail = check_refusal(out.stop_reason, item)
        elif item["check"] == "ragas":
            result.ragas_scores = await score_ragas(
                ragas_metrics, item["question"], out.answer, out.contexts, item["ground_truth"]
            )
            result.detail = _detail_for_ragas(result.ragas_scores)

        print(f"  -> {result.detail}")
        results.append(result)

    return results


def print_summary(results: list[QuestionResult], *, pipeline: str) -> None:
    print(f"\n=== Summary ({pipeline} pipeline, {len(results)} questions) ===\n")

    errored = [r for r in results if r.error]
    if errored:
        print(f"Pipeline errors (excluded from scoring below): {len(errored)}/{len(results)}")
        for r in errored:
            print(f"  ERROR [{r.id}] {r.question!r} - {r.detail}")

    pass_fail = [r for r in results if r.passed is not None]
    if pass_fail:
        passed = sum(1 for r in pass_fail if r.passed)
        print(f"Pass/fail checks: {passed}/{len(pass_fail)} passed")
        for r in pass_fail:
            if not r.passed:
                print(f"  FAIL [{r.id}] {r.question!r} - {r.detail}")

    ragas_results = [r for r in results if r.ragas_scores is not None]
    if ragas_results:
        print("\nRagas averages (grounded questions):")
        for name in RAGAS_METRIC_NAMES:
            values = [r.ragas_scores[name] for r in ragas_results if r.ragas_scores.get(name) is not None]
            avg = sum(values) / len(values) if values else None
            print(f"  {name:<18} {avg:.3f}" if avg is not None else f"  {name:<18} n/a")


def save_results(results: list[QuestionResult], *, pipeline: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"{pipeline}_{timestamp}.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

    md_path = RESULTS_DIR / f"{pipeline}_latest.md"
    md_path.write_text(_render_markdown(results, pipeline=pipeline), encoding="utf-8")
    return json_path


def _render_markdown(results: list[QuestionResult], *, pipeline: str) -> str:
    lines = [f"# Eval results - {pipeline} pipeline\n"]

    errored = [r for r in results if r.error]
    if errored:
        lines.append(f"## Pipeline errors: {len(errored)}/{len(results)}\n")
        lines.append("| ID | Question | Error |")
        lines.append("|---|---|---|")
        for r in errored:
            lines.append(f"| {r.id} | {r.question} | {r.detail} |")
        lines.append("")

    ragas_results = [r for r in results if r.ragas_scores is not None]
    if ragas_results:
        lines.append("## Ragas averages (grounded questions)\n")
        lines.append("| Metric | Average |")
        lines.append("|---|---|")
        for name in RAGAS_METRIC_NAMES:
            values = [r.ragas_scores[name] for r in ragas_results if r.ragas_scores.get(name) is not None]
            avg = f"{sum(values) / len(values):.3f}" if values else "n/a"
            lines.append(f"| {name} | {avg} |")
        lines.append("")

    pass_fail = [r for r in results if r.passed is not None]
    if pass_fail:
        passed = sum(1 for r in pass_fail if r.passed)
        lines.append(f"## Pass/fail checks: {passed}/{len(pass_fail)}\n")
        lines.append("| ID | Category | Question | Result |")
        lines.append("|---|---|---|---|")
        for r in pass_fail:
            status = "PASS" if r.passed else f"FAIL ({r.detail})"
            lines.append(f"| {r.id} | {r.category} | {r.question} | {status} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", choices=["improved", "naive"], default="improved")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N matching questions")
    parser.add_argument("--ids", default=None, help="comma-separated question ids, e.g. f01,f02,n01")
    parser.add_argument("--category", default=None, help="run only one category")
    args = parser.parse_args()

    results = asyncio.run(run_all(args))
    if not results:
        return 1

    print_summary(results, pipeline=args.pipeline)
    path = save_results(results, pipeline=args.pipeline)
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
