import asyncio

import pytest

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.evaluate.console_report import EvaluationConsoleReport
from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_trace


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("use_iterator", [False, True])
@pytest.mark.parametrize("file_type", ["md", "html"])
def test_export_without_print_results(
    tmp_path, capsys, monkeypatch, run_async, use_iterator, file_type
):
    def unexpected_render(*args, **kwargs):
        pytest.fail("Quiet export must not render the terminal report")

    monkeypatch.setattr(
        EvaluationConsoleReport, "render_to_terminal", unexpected_render
    )
    target = tmp_path / "reports"
    kwargs = {
        "metrics": [ExactMatchMetric()],
        "identifier": "quiet_export",
        "async_config": AsyncConfig(run_async=run_async),
        "cache_config": CacheConfig(write_cache=False, use_cache=False),
        "display_config": DisplayConfig(
            print_results=False,
            show_indicator=False,
            file_type=file_type,
            file_output_dir=str(target),
            inspect_after_run=False,
        ),
    }

    if use_iterator:

        @observe()
        def app(value):
            update_current_trace(
                input=value, output="Paris", expected_output="Paris"
            )

        @observe()
        async def async_app(value):
            update_current_trace(
                input=value, output="Paris", expected_output="Paris"
            )

        dataset = EvaluationDataset(
            goldens=[
                Golden(input="Capital of France?", expected_output="Paris")
            ]
        )
        for golden in dataset.evals_iterator(**kwargs):
            if run_async:
                dataset.evaluate(asyncio.create_task(async_app(golden.input)))
            else:
                app(golden.input)
    else:
        evaluate(
            test_cases=[
                LLMTestCase(
                    input="Capital of France?",
                    actual_output="Paris",
                    expected_output="Paris",
                )
            ],
            **kwargs,
        )

    files = list(target.glob(f"quiet_export_*.{file_type}"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "DeepEval Evaluation Results" in content
    assert "Exact Match" in content
    assert "Capital of France?" in content
    assert "DeepEval Evaluation Results" not in capsys.readouterr().out
