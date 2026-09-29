import asyncio
import json
from types import SimpleNamespace

import pytest

from research_agent.agent_tools import framework  # noqa: F401  建立 tensor_inpainting_agent 命名空间

from tensor_inpainting_agent.core.llm_adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    OpenAIResponsesAdapter,
    create_adapter,
)


# ==================== 适配器选择 ====================


def test_create_adapter_defaults_to_chat_completions(monkeypatch):
    monkeypatch.delenv("LLM_API_STYLE", raising=False)
    adapter = create_adapter("key", "https://api.example.com/v1", 60, "gpt-4o")
    assert isinstance(adapter, OpenAIAdapter)


def test_create_adapter_selects_responses_via_env(monkeypatch):
    monkeypatch.setenv("LLM_API_STYLE", "responses")
    adapter = create_adapter("key", "https://api.openai.com/v1", 60, "gpt-4o")
    assert isinstance(adapter, OpenAIResponsesAdapter)


def test_create_adapter_selects_responses_via_argument(monkeypatch):
    monkeypatch.delenv("LLM_API_STYLE", raising=False)
    adapter = create_adapter(
        "key", "https://api.openai.com/v1", 60, "gpt-4o", api_style="responses"
    )
    assert isinstance(adapter, OpenAIResponsesAdapter)


def test_api_style_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_API_STYLE", "responses")
    adapter = create_adapter(
        "key", "https://api.openai.com/v1", 60, "gpt-4o", api_style="chat"
    )
    assert isinstance(adapter, OpenAIAdapter)


def test_other_providers_ignore_api_style(monkeypatch):
    monkeypatch.setenv("LLM_API_STYLE", "responses")
    assert isinstance(
        create_adapter("key", "https://api.anthropic.com", 60, "claude-sonnet-5"),
        AnthropicAdapter,
    )
    assert isinstance(
        create_adapter(
            "key", "https://generativelanguage.googleapis.com", 60, "gemini-2.5-pro"
        ),
        GeminiAdapter,
    )


# ==================== 请求格式转换 ====================


@pytest.fixture
def adapter():
    return OpenAIResponsesAdapter("key", "https://api.openai.com/v1", 60, "gpt-4o")


def test_system_message_becomes_instructions(adapter):
    request = adapter._build_request(
        [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ]
    )
    assert request["instructions"] == "你是助手"
    assert request["input"] == [{"role": "user", "content": "你好"}]


def test_user_message_stays_in_input_not_instructions(adapter):
    request = adapter._build_request([{"role": "user", "content": "你好"}])
    assert "instructions" not in request
    assert request["input"] == [{"role": "user", "content": "你好"}]


def test_tool_result_becomes_function_call_output(adapter):
    request = adapter._build_request(
        [{"role": "tool", "tool_call_id": "call_1", "content": "结果"}]
    )
    assert request["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "结果"}
    ]


def test_non_string_tool_result_is_serialized(adapter):
    request = adapter._build_request(
        [{"role": "tool", "tool_call_id": "call_1", "content": {"value": 1}}]
    )
    assert json.loads(request["input"][0]["output"]) == {"value": 1}


def test_assistant_tool_calls_become_function_call_items(adapter):
    request = adapter._build_request(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "x"}'},
                    }
                ],
            }
        ]
    )
    assert request["input"] == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "search",
            "arguments": '{"q": "x"}',
        }
    ]


def test_assistant_with_text_and_tool_calls_emits_both(adapter):
    request = adapter._build_request(
        [
            {
                "role": "assistant",
                "content": "我先查一下",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            }
        ]
    )
    assert request["input"][0] == {"role": "assistant", "content": "我先查一下"}
    assert request["input"][1]["type"] == "function_call"


def test_multimodal_blocks_are_converted(adapter):
    request = adapter._build_request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看图"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AAA", "detail": "high"},
                    },
                ],
            }
        ]
    )
    assert request["input"][0]["content"] == [
        {"type": "input_text", "text": "看图"},
        {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,AAA",
            "detail": "high",
        },
    ]


def test_max_tokens_is_renamed(adapter):
    request = adapter._build_request([{"role": "user", "content": "hi"}], max_tokens=256)
    assert request["max_output_tokens"] == 256
    assert "max_tokens" not in request


def test_max_output_tokens_passes_through(adapter):
    request = adapter._build_request(
        [{"role": "user", "content": "hi"}], max_output_tokens=512
    )
    assert request["max_output_tokens"] == 512


def test_none_valued_kwargs_are_dropped(adapter):
    request = adapter._build_request([{"role": "user", "content": "hi"}], temperature=None)
    assert "temperature" not in request


def test_tools_are_flattened(adapter):
    tools = adapter._convert_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "搜索",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
    )
    assert tools == [
        {
            "type": "function",
            "name": "search",
            "description": "搜索",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]


def test_tool_choice_is_flattened(adapter):
    converted = adapter._convert_tool_choice(
        {"type": "function", "function": {"name": "search"}}
    )
    assert converted == {"type": "function", "name": "search"}
    assert adapter._convert_tool_choice("auto") == "auto"
    assert adapter._convert_tool_choice("required") == "required"


# ==================== 响应解析 ====================


class FakeResponses:
    def __init__(self, response=None, events=None):
        self.response = response
        self.events = events or []
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.events)
        return self.response


class FakeAsyncResponses(FakeResponses):
    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs.get("stream"):
            return _async_iter(self.events)
        return self.response


class _async_iter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses


def _text_response(text, model="gpt-4o"):
    return SimpleNamespace(
        model=model,
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _attach(adapter, responses):
    adapter._client = FakeClient(responses)
    return adapter


def test_invoke_parses_text_and_usage(adapter):
    responses = FakeResponses(_text_response("你好"))
    _attach(adapter, responses)

    result = adapter.invoke([{"role": "user", "content": "hi"}])

    assert result.content == "你好"
    assert result.model == "gpt-4o"
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert result.latency_ms >= 0


def test_invoke_parses_reasoning_summary(adapter):
    responses = FakeResponses(
        SimpleNamespace(
            model="gpt-4o",
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="先分析")],
                ),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="答案")],
                ),
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3),
        )
    )
    _attach(adapter, responses)

    result = adapter.invoke([{"role": "user", "content": "hi"}])

    assert result.content == "答案"
    assert result.reasoning_content == "先分析"


def test_invoke_with_tools_parses_function_call(adapter):
    responses = FakeResponses(
        SimpleNamespace(
            model="gpt-4o",
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_abc",
                    name="search",
                    arguments='{"q": "tensor"}',
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3),
        )
    )
    _attach(adapter, responses)

    result = adapter.invoke_with_tools(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "search"}}],
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_abc"
    assert result.tool_calls[0].name == "search"
    assert json.loads(result.tool_calls[0].arguments) == {"q": "tensor"}
    assert responses.requests[0]["tools"] == [
        {"type": "function", "name": "search", "description": "", "parameters": {"type": "object", "properties": {}}}
    ]
    assert responses.requests[0]["tool_choice"] == "auto"


def test_stream_invoke_yields_deltas_and_records_usage(adapter):
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="你"),
        SimpleNamespace(type="response.output_text.delta", delta="好"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=7, output_tokens=3, total_tokens=10)
            ),
        ),
    ]
    responses = FakeResponses(events=events)
    _attach(adapter, responses)

    chunks = list(adapter.stream_invoke([{"role": "user", "content": "hi"}]))

    assert chunks == ["你", "好"]
    assert adapter.last_stats.usage["total_tokens"] == 10


def test_stream_invoke_collects_reasoning_summary(adapter):
    events = [
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="思考中"),
        SimpleNamespace(type="response.output_text.delta", delta="答案"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
            ),
        ),
    ]
    _attach(adapter, FakeResponses(events=events))

    assert list(adapter.stream_invoke([{"role": "user", "content": "hi"}])) == ["答案"]
    assert adapter.last_stats.reasoning_content == "思考中"


def test_astream_invoke_yields_deltas(adapter):
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="异"),
        SimpleNamespace(type="response.output_text.delta", delta="步"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
            ),
        ),
    ]
    adapter._async_client = FakeClient(FakeAsyncResponses(events=events))

    async def consume():
        return [chunk async for chunk in adapter.astream_invoke([{"role": "user", "content": "hi"}])]

    assert asyncio.run(consume()) == ["异", "步"]
    assert adapter.last_stats.usage["total_tokens"] == 2


def test_errors_are_wrapped(adapter):
    from tensor_inpainting_agent.core.exceptions import TensorInpaintingException

    class BoomResponses:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    adapter._client = FakeClient(BoomResponses())

    with pytest.raises(TensorInpaintingException):
        adapter.invoke([{"role": "user", "content": "hi"}])


# ==================== 失败状态与多轮状态回传 ====================


@pytest.mark.parametrize("method", ["invoke", "invoke_with_tools"])
@pytest.mark.parametrize("status,reason", [
    ("failed", "server unavailable"),
    ("incomplete", "max_output_tokens"),
    ("cancelled", "cancelled"),
    ("in_progress", "in_progress"),
])
def test_noncompleted_response_raises_instead_of_returning_partial_output(adapter, method, status, reason):
    from tensor_inpainting_agent.core.exceptions import TensorInpaintingException

    response = _text_response("partial JSON: {")
    response.status = status
    response.error = SimpleNamespace(message=reason) if status == "failed" else None
    response.incomplete_details = SimpleNamespace(reason=reason) if status == "incomplete" else None
    _attach(adapter, FakeResponses(response))
    kwargs = {"tools": []} if method == "invoke_with_tools" else {}
    with pytest.raises(TensorInpaintingException, match=reason):
        getattr(adapter, method)([{"role": "user", "content": "hi"}], **kwargs)


class ClosingStream:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.events)

    def close(self):
        self.closed = True


class ClosingAsyncStream(ClosingStream):
    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


@pytest.mark.parametrize("mode", ["stream_invoke", "think", "astream_invoke"])
@pytest.mark.parametrize("terminal,reason", [
    ("response.failed", "server unavailable"),
    ("response.incomplete", "max_output_tokens"),
    ("error", "stream error"),
    (None, "未收到 response.completed"),
    ("response.completed", None),
])
def test_stream_terminal_states_close_stream_and_clear_stale_stats(adapter, mode, terminal, reason):
    from tensor_inpainting_agent.core.exceptions import TensorInpaintingException
    from tensor_inpainting_agent.core.llm import TensorInpaintingLLM

    events = [SimpleNamespace(type="response.output_text.delta", delta="partial")]
    response = _text_response("partial")
    response.status = terminal.split(".")[-1] if terminal else None
    response.error = SimpleNamespace(message=reason) if terminal == "response.failed" else None
    response.incomplete_details = SimpleNamespace(reason=reason)
    if terminal:
        events.append(SimpleNamespace(type=terminal, response=response, message=reason))
    stream = ClosingAsyncStream(events) if mode == "astream_invoke" else ClosingStream(events)

    async def async_create(**kwargs):
        return stream

    backend = SimpleNamespace(create=async_create if mode == "astream_invoke" else lambda **kwargs: stream)
    adapter._client = FakeClient(backend)
    adapter._async_client = FakeClient(backend)
    llm = TensorInpaintingLLM(model="gpt-4o", api_key="key", base_url="https://example.invalid/v1", api_style="responses")
    llm._adapter = adapter
    llm.last_call_stats = adapter.last_stats = object()
    chunks = []

    async def consume_async():
        async for chunk in llm.astream_invoke([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

    def consume():
        if mode == "astream_invoke":
            asyncio.run(consume_async())
        else:
            chunks.extend(getattr(llm, mode)([{"role": "user", "content": "hi"}]))

    if reason:
        with pytest.raises(TensorInpaintingException, match=reason):
            consume()
        assert adapter.last_stats is None
        assert llm.last_call_stats is None
    else:
        consume()
        assert llm.last_call_stats.usage["total_tokens"] == 15
    assert chunks == ["partial"]
    assert stream.closed


@pytest.mark.parametrize("agent_kind", ["simple", "react"])
def test_agent_tool_round_trip_preserves_raw_responses_output(agent_kind):
    from datetime import datetime
    from pydantic import BaseModel, ConfigDict
    from tensor_inpainting_agent.agents.simple_agent import SimpleAgent
    from tensor_inpainting_agent.agents.react_agent import ReActAgent
    from tensor_inpainting_agent.core.llm import TensorInpaintingLLM
    from tensor_inpainting_agent.tools.builtin.calculator import CalculatorTool
    from tensor_inpainting_agent.tools.registry import ToolRegistry

    # Pydantic 输出条目模拟 SDK model_dump；文本 phase、推理密文与调用 ID 均需原样回传。
    class OutputItem(BaseModel):
        model_config = ConfigDict(extra="allow")
        type: str

    raw_items = [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-state"},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "phase": "commentary", "content": [{"type": "output_text", "text": "Calculating", "annotations": []}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "python_calculator",
         "arguments": '{"input":"1+1"}', "status": "completed"},
    ]
    first = SimpleNamespace(status="completed", model="gpt-4o", usage=None,
                            output=[OutputItem(**item) for item in raw_items])

    class TwoRoundResponses(FakeResponses):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            return first if len(self.requests) == 1 else _text_response("2")

    backend = TwoRoundResponses()
    llm = TensorInpaintingLLM(model="gpt-4o", api_key="key", base_url="https://example.invalid/v1", api_style="responses")
    llm._adapter._client = FakeClient(backend)
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    agent_type = SimpleAgent if agent_kind == "simple" else ReActAgent
    # 直接测试真实工具循环，隔离历史压缩、持久化及其可选依赖。
    agent = agent_type.__new__(agent_type)
    agent.name = "test"
    agent.llm = llm
    agent.tool_registry = registry
    agent.config = SimpleNamespace(trace_enabled=False)
    agent.trace_logger = None
    agent.enable_tool_calling = True
    agent.max_tool_iterations = agent.max_steps = 3
    agent._builtin_tools = {}
    agent._build_messages = lambda text: [{"role": "user", "content": text}]
    agent._build_tool_schemas = lambda: [CalculatorTool().to_openai_schema()]
    agent.add_message = lambda message: None
    agent._execute_tool_call = lambda name, arguments: CalculatorTool().run(arguments).text
    kwargs = {"store": False, "include": ["message.output_text.logprobs"]}
    if agent_kind == "simple":
        answer = agent.run("1+1?", **kwargs)
    else:
        answer = agent._run_impl("1+1?", datetime.now(), **kwargs)
    assert answer == "2"
    assert len(backend.requests) == 2
    request = backend.requests[1]
    assert request["input"][1:4] == raw_items
    assert len(request["input"]) == 5
    assert request["input"][4]["type"] == "function_call_output"
    assert request["input"][4]["call_id"] == "call_1"
    assert "2" in request["input"][4]["output"]
    assert request["store"] is False
    assert request["include"] == ["message.output_text.logprobs", "reasoning.encrypted_content"]


def test_chat_tool_history_keeps_existing_wire_format():
    from tensor_inpainting_agent.core.llm_response import LLMToolResponse, ToolCall

    result = LLMToolResponse(content=None, model="test", tool_calls=[ToolCall("call_1", "search", "{}")])
    assert result.to_assistant_message() == {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
    }
