from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.run_eval import (
    GOLDEN_SET_PATH,
    PipelineOutput,
    build_ragas_metrics,
    check_keyword,
    check_refusal,
    load_golden_set,
    run_all,
    run_improved,
    run_naive,
    score_ragas,
)

# ---- golden_set.yaml structural integrity - catches authoring mistakes ----

_VALID_CATEGORIES = {"factual", "no_answer", "adversarial", "conflicting_source", "supersedes", "table"}
_VALID_CHECKS = {"ragas", "keyword", "refusal"}


def test_golden_set_file_exists():
    assert GOLDEN_SET_PATH.exists()


def test_golden_set_has_30_to_50_questions():
    questions = load_golden_set()
    assert 30 <= len(questions) <= 50


def test_golden_set_ids_are_unique():
    questions = load_golden_set()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_golden_set_categories_and_checks_are_from_the_known_set():
    for q in load_golden_set():
        assert q["category"] in _VALID_CATEGORIES, q
        assert q["check"] in _VALID_CHECKS, q


def test_ragas_questions_have_ground_truth():
    for q in load_golden_set():
        if q["check"] == "ragas":
            assert q.get("ground_truth"), q


def test_refusal_questions_have_expected_stop_reasons():
    for q in load_golden_set():
        if q["check"] == "refusal":
            assert q.get("expected_stop_reasons"), q


def test_keyword_questions_have_at_least_one_phrase_list():
    for q in load_golden_set():
        if q["check"] == "keyword":
            assert q.get("expected_phrases") or q.get("forbidden_phrases"), q


def test_every_category_is_represented():
    categories = {q["category"] for q in load_golden_set()}
    assert categories == _VALID_CATEGORIES


# ---- check_keyword / check_refusal ----


def test_check_keyword_passes_when_expected_present_and_none_forbidden():
    passed, _ = check_keyword(
        "The documents disagree: 48 hours vs 3-5 business days.",
        {"expected_phrases": ["48 hours", "3-5 business days"], "forbidden_phrases": ["9999"]},
    )
    assert passed is True


def test_check_keyword_fails_when_expected_phrase_missing():
    passed, detail = check_keyword("Some unrelated answer.", {"expected_phrases": ["48 hours"]})
    assert passed is False
    assert "48 hours" in detail


def test_check_keyword_fails_when_forbidden_phrase_present():
    passed, detail = check_keyword(
        "All refunds take 9999 days.", {"forbidden_phrases": ["9999"]}
    )
    assert passed is False
    assert "9999" in detail


def test_check_keyword_is_case_insensitive():
    passed, _ = check_keyword("THE COST IS $8.00", {"expected_phrases": ["$8.00"]})
    assert passed is True


def test_check_refusal_passes_when_stop_reason_matches():
    passed, _ = check_refusal("no_context", {"expected_stop_reasons": ["no_context", "low_confidence"]})
    assert passed is True


def test_check_refusal_fails_when_stop_reason_is_end_turn():
    passed, _ = check_refusal("end_turn", {"expected_stop_reasons": ["no_context", "low_confidence"]})
    assert passed is False


def test_check_refusal_fails_for_naive_pipeline_which_never_refuses():
    # naive_generate has no stop_reason concept - run_naive always reports None.
    passed, _ = check_refusal(None, {"expected_stop_reasons": ["no_context", "low_confidence"]})
    assert passed is False


# ---- run_improved / run_naive wiring ----


class _FakeEmbedder:
    def embed(self, texts, input_type):
        return [[0.1] for _ in texts]


class _FakeChunk:
    def __init__(self, text):
        self.chunk_text = text


def test_run_improved_maps_chunks_and_answer_correctly(monkeypatch):
    import scripts.run_eval as run_eval_module

    fake_chunks = [_FakeChunk("context A"), _FakeChunk("context B")]
    monkeypatch.setattr(run_eval_module, "search", lambda *a, **k: fake_chunks)
    monkeypatch.setattr(
        run_eval_module,
        "generate_answer",
        lambda *a, **k: SimpleNamespace(text="the answer", stop_reason="end_turn"),
    )

    out = run_improved(
        "q", settings=object(), embedder=_FakeEmbedder(), reranker=object(), anthropic_client=object()
    )

    assert out.answer == "the answer"
    assert out.contexts == ["context A", "context B"]
    assert out.stop_reason == "end_turn"


def test_run_naive_always_reports_none_stop_reason(monkeypatch):
    import scripts.run_eval as run_eval_module

    monkeypatch.setattr(run_eval_module, "naive_search", lambda *a, **k: ["ctx"])
    monkeypatch.setattr(run_eval_module, "naive_generate", lambda *a, **k: "naive answer")

    out = run_naive("q", settings=object(), embedder=_FakeEmbedder(), anthropic_client=object())

    assert out.answer == "naive answer"
    assert out.contexts == ["ctx"]
    assert out.stop_reason is None


# ---- score_ragas ----


@pytest.mark.asyncio
async def test_score_ragas_returns_none_scores_when_no_contexts():
    scores = await score_ragas({}, "q", "a", [], "ground truth")
    assert scores == {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }


class _FakeMetricResult:
    def __init__(self, value):
        self.value = value


class _FakeRagasMetric:
    def __init__(self, value):
        self._value = value

    async def ascore(self, **kwargs):
        return _FakeMetricResult(self._value)


@pytest.mark.asyncio
async def test_score_ragas_calls_all_four_metrics_and_collects_their_values():
    metrics = {
        "faithfulness": _FakeRagasMetric(0.9),
        "answer_relevancy": _FakeRagasMetric(0.8),
        "context_precision": _FakeRagasMetric(0.7),
        "context_recall": _FakeRagasMetric(0.6),
    }

    scores = await score_ragas(metrics, "q", "a", ["some context"], "ground truth")

    assert scores == {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.6,
    }


class _FailingRagasMetric:
    async def ascore(self, **kwargs):
        raise RuntimeError("rate limited")


@pytest.mark.asyncio
async def test_score_ragas_one_metric_failing_does_not_sink_the_others():
    metrics = {
        "faithfulness": _FailingRagasMetric(),
        "answer_relevancy": _FakeRagasMetric(0.8),
        "context_precision": _FakeRagasMetric(0.7),
        "context_recall": _FakeRagasMetric(0.6),
    }

    scores = await score_ragas(metrics, "q", "a", ["some context"], "ground truth")

    assert scores["faithfulness"] is None
    assert scores["answer_relevancy"] == 0.8


# ---- build_ragas_metrics: claude-sonnet-5 dropped temperature/top_p ----


def test_build_ragas_metrics_strips_temperature_and_top_p():
    from app.config import Settings

    settings = Settings(
        anthropic_api_key="test-key",
        voyage_api_key="test-key",
        generation_model="claude-sonnet-5",
        embedding_model="voyage-4-lite",
    )

    metrics = build_ragas_metrics(settings)

    for name, metric in metrics.items():
        assert "temperature" not in metric.llm.model_args, name
        assert "top_p" not in metric.llm.model_args, name


# ---- run_all: per-question failure isolation ----


@pytest.mark.asyncio
async def test_run_all_isolates_a_failing_question_and_continues(monkeypatch):
    import scripts.run_eval as run_eval_module

    fake_questions = [
        {"id": "q1", "category": "factual", "check": "keyword", "question": "will fail", "expected_phrases": ["x"]},
        {"id": "q2", "category": "factual", "check": "keyword", "question": "will pass", "expected_phrases": ["ok"]},
    ]
    monkeypatch.setattr(run_eval_module, "load_golden_set", lambda: fake_questions)
    monkeypatch.setattr(run_eval_module, "get_settings", lambda: SimpleNamespace(anthropic_api_key="k"))
    monkeypatch.setattr(run_eval_module, "make_anthropic_client", lambda *a, **k: object())
    monkeypatch.setattr(run_eval_module, "get_embedding_provider", lambda *a, **k: object())
    monkeypatch.setattr(run_eval_module, "get_rerank_provider", lambda *a, **k: object())
    monkeypatch.setattr(run_eval_module, "build_ragas_metrics", lambda *a, **k: {})

    call_count = {"n": 0}

    def fake_run_improved(question, **kwargs):
        call_count["n"] += 1
        if question == "will fail":
            raise RuntimeError("rate limited")
        return PipelineOutput(answer="ok", contexts=["ctx"], stop_reason="end_turn")

    monkeypatch.setattr(run_eval_module, "run_improved", fake_run_improved)

    args = SimpleNamespace(ids=None, category=None, limit=None, pipeline="improved")
    results = await run_all(args)

    assert call_count["n"] == 2  # both questions attempted despite the first failing
    assert len(results) == 2
    assert results[0].error is True
    assert "rate limited" in results[0].detail
    assert results[1].error is False
    assert results[1].passed is True
