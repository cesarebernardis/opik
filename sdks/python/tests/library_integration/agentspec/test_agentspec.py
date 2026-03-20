from collections.abc import Iterator

import opik
import pytest

pytest.importorskip("pyagentspec")

from opik.integrations.agentspec import AgentSpecInstrumentor, OpikSpanProcessor
from pyagentspec.llms import OpenAiConfig
from pyagentspec.tools import ClientTool
from pyagentspec.tracing.events import (
    LlmGenerationRequest,
    LlmGenerationResponse,
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from pyagentspec.tracing.messages.message import Message
from pyagentspec.tracing.spans import LlmGenerationSpan, ToolExecutionSpan
from pyagentspec.tracing.trace import Trace, get_trace


def _iter_spans(spans) -> Iterator:
    for span in spans:
        yield span
        yield from _iter_spans(span.spans)


def _find_span(trace_tree, name: str):
    for span in _iter_spans(trace_tree.spans):
        if span.name == name:
            return span
    raise AssertionError(f"Span {name!r} was not recorded")


def test_opik_span_processor__tool_and_llm_spans_are_forwarded_to_opik(
    fake_backend,
):
    project_name = "agentspec-integration-test"
    tool = ClientTool(name="lookup_weather")
    llm_config = OpenAiConfig(name="demo-model", model_id="gpt-4o-mini")
    span_processor = OpikSpanProcessor(
        project_name=project_name,
        mask_sensitive_information=False,
    )

    with Trace(name="AgentSpec workflow", span_processors=[span_processor]):
        with ToolExecutionSpan(
            name="weather_tool",
            tool=tool,
            events=[
                ToolExecutionRequest(
                    tool=tool,
                    inputs={"city": "Zurich"},
                    request_id="tool-request",
                ),
                ToolExecutionResponse(
                    tool=tool,
                    outputs={"temperature": "18C"},
                    request_id="tool-request",
                ),
            ],
        ):
            pass

        with LlmGenerationSpan(
            name="llm_generation",
            llm_config=llm_config,
            events=[
                LlmGenerationRequest(
                    llm_config=llm_config,
                    prompt=[Message(content="my prompt", role="system", sender="me")],
                    tools=[],
                    request_id="llm-request",
                ),
                LlmGenerationResponse(
                    llm_config=llm_config,
                    content="sunny",
                    request_id="llm-request",
                    input_tokens=11,
                    output_tokens=4,
                ),
            ],
        ):
            pass

    opik.flush_tracker()

    assert len(fake_backend.trace_trees) == 1

    trace_tree = fake_backend.trace_trees[0]
    assert trace_tree.name == "AgentSpec workflow"
    assert trace_tree.project_name == project_name

    tool_span = _find_span(trace_tree, "weather_tool")
    assert tool_span.type == "tool"
    assert tool_span.project_name == project_name
    assert tool_span.input == {"city": "Zurich"}
    assert tool_span.output == {"temperature": "18C"}
    assert len(tool_span.metadata["events"]) == 2

    llm_span = _find_span(trace_tree, "llm_generation")
    assert llm_span.type == "llm"
    assert llm_span.project_name == project_name
    assert llm_span.model == "demo-model"
    assert llm_span.input["request_id"] == "llm-request"
    assert llm_span.input["prompt"] == [
        {"id": None, "content": "my prompt", "role": "system", "sender": "me"}
    ]
    assert llm_span.output["response"] == "sunny"
    assert llm_span.output["tool_calls"] == []
    assert llm_span.output["completion_id"] is None
    assert len(llm_span.metadata["events"]) == 2


def test_agentspec_instrumentor__context_manager_records_spans_and_cleans_up(
    fake_backend,
):
    project_name = "agentspec-instrumentor-test"
    tool = ClientTool(name="lookup_time")
    instrumentor = AgentSpecInstrumentor()

    with instrumentor.instrument_context(
        project_name=project_name,
        mask_sensitive_information=False,
    ):
        assert get_trace() is not None

        with ToolExecutionSpan(
            name="time_tool",
            tool=tool,
            events=[
                ToolExecutionRequest(
                    tool=tool,
                    inputs={"timezone": "Europe/Zurich"},
                    request_id="tool-request",
                ),
                ToolExecutionResponse(
                    tool=tool,
                    outputs={"time": "09:30"},
                    request_id="tool-request",
                ),
            ],
        ):
            pass

    opik.flush_tracker()

    assert get_trace() is None
    assert len(fake_backend.trace_trees) == 1

    trace_tree = fake_backend.trace_trees[0]
    assert trace_tree.project_name == project_name

    tool_span = _find_span(trace_tree, "time_tool")
    assert tool_span.type == "tool"
    assert tool_span.input == {"timezone": "Europe/Zurich"}
    assert tool_span.output == {"time": "09:30"}


def test_agentspec_instrumentor__active_trace_exists__raises_value_error():
    instrumentor = AgentSpecInstrumentor()

    with Trace(name="existing trace"):
        with pytest.raises(
            ValueError,
            match="Agent Spec Trace already active",
        ):
            instrumentor.instrument(project_name="agentspec-instrumentor-test")
