"""LLM适配器 - 支持OpenAI、Anthropic、Gemini等不同接口格式"""

import os
import time
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Optional, Iterator, List, Dict, Any, Union, AsyncIterator

from .llm_response import LLMResponse, StreamStats, LLMToolResponse, ToolCall
from .exceptions import TensorInpaintingException


class BaseLLMAdapter(ABC):
    """LLM适配器基类"""

    def __init__(self, api_key: str, base_url: Optional[str], timeout: int, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        self._client = None
        self._async_client = None

    @abstractmethod
    def create_client(self) -> Any:
        """创建客户端实例"""
        pass

    def create_async_client(self) -> Any:
        """创建异步客户端实例（子类可选实现）"""
        return None

    @abstractmethod
    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        pass

    @abstractmethod
    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用，返回生成器"""
        pass

    async def astream_invoke(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        """异步流式调用（子类可选实现真正的异步）

        默认实现：使用队列 + 线程池包装同步流式方法
        """
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream_to_queue():
            try:
                for chunk in self.stream_invoke(messages, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # 在线程池中运行同步流式方法
        loop.run_in_executor(None, _stream_to_queue)

        # 从队列中逐个取出 chunk
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    @abstractmethod
    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> LLMToolResponse:
        """工具调用（Function Calling）"""
        pass

    def _is_thinking_model(self, model_name: str) -> bool:
        """判断是否为thinking model"""
        thinking_keywords = ["reasoner", "o1", "o3", "thinking"]
        model_lower = model_name.lower()
        return any(keyword in model_lower for keyword in thinking_keywords)


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI Chat Completions 适配器（/v1/chat/completions，默认）

    支持：
    - OpenAI官方API
    - 所有OpenAI兼容接口（DeepSeek、Qwen、Kimi、智谱等）
    - Thinking Models（o1、deepseek-reasoner等）

    若需要 OpenAI 官方新的 Responses API，请改用 OpenAIResponsesAdapter
    （设置 LLM_API_STYLE=responses）。
    """

    def create_client(self) -> Any:
        """创建OpenAI客户端"""
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def create_async_client(self) -> Any:
        """创建OpenAI异步客户端"""
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()
        
        start_time = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # 提取内容和推理过程
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning_content = None
            
            # Thinking model特殊处理
            if self._is_thinking_model(self.model):
                # OpenAI o1系列：reasoning_content在message中
                if hasattr(choice.message, 'reasoning_content'):
                    reasoning_content = choice.message.reasoning_content
                # DeepSeek reasoner：可能在其他字段
                elif hasattr(choice, 'reasoning_content'):
                    reasoning_content = choice.reasoning_content
            
            # 提取usage信息
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )
            
        except Exception as e:
            raise TensorInpaintingException(f"OpenAI API调用失败: {str(e)}")
    
    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()
        
        start_time = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            collected_content = []
            reasoning_content = None
            usage = {}
            
            for chunk in response:
                choices = getattr(chunk, "choices", None)
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        # 提取内容
                        content = getattr(delta, "content", None)
                        if content:
                            collected_content.append(content)
                            yield content

                        # Thinking model的推理过程
                        if self._is_thinking_model(self.model):
                            reasoning_delta = getattr(delta, "reasoning_content", None)
                            if reasoning_delta:
                                if reasoning_content is None:
                                    reasoning_content = ""
                                reasoning_content += reasoning_delta

                # 提取usage（流式最后一个chunk可能包含）
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            # 返回统计信息（存储到适配器，供外部获取）
            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )

        except Exception as e:
            raise TensorInpaintingException(f"OpenAI API流式调用失败: {str(e)}")

    async def astream_invoke(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        """真正的异步流式调用（使用 OpenAI 原生异步客户端）"""
        if not self._async_client:
            self._async_client = self.create_async_client()

        start_time = time.time()

        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )

            collected_content = []
            reasoning_content = None
            usage = {}

            async for chunk in response:
                choices = getattr(chunk, "choices", None)
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        # 提取内容
                        content = getattr(delta, "content", None)
                        if content:
                            collected_content.append(content)
                            yield content

                        # Thinking model的推理过程
                        if self._is_thinking_model(self.model):
                            reasoning_delta = getattr(delta, "reasoning_content", None)
                            if reasoning_delta:
                                if reasoning_content is None:
                                    reasoning_content = ""
                                reasoning_content += reasoning_delta

                # 提取usage（流式最后一个chunk可能包含）
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            # 返回统计信息（存储到适配器，供外部获取）
            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )

        except Exception as e:
            raise TensorInpaintingException(f"OpenAI API异步流式调用失败: {str(e)}")

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict],
                         tool_choice: Union[str, Dict] = "auto", **kwargs) -> LLMToolResponse:
        """工具调用（Function Calling）"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs
            )

            latency_ms = int((time.time() - start_time) * 1000)
            message = response.choices[0].message

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments
                    ))

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return LLMToolResponse(
                content=message.content,
                tool_calls=tool_calls,
                model=response.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise TensorInpaintingException(f"OpenAI Function Calling调用失败: {str(e)}")


class OpenAIResponsesAdapter(BaseLLMAdapter):
    """OpenAI Responses API 适配器（/v1/responses）

    与 OpenAIAdapter（Chat Completions）的关键差异：
    - 端点：responses.create 而非 chat.completions.create
    - system 消息通过顶层 instructions 参数传入，不放进 input 列表
    - 消息放在 input 中；工具调用与结果用 function_call / function_call_output 条目表达
    - 工具 schema 为扁平结构（name/description/parameters 与 type 同级，无 function 包裹）
    - 上限参数为 max_output_tokens，需要与 max_tokens 互相映射
    - 多模态图片块为 input_image，image_url 直接是字符串

    注意：/v1/responses 目前主要由 OpenAI 官方提供。DeepSeek、Qwen、Kimi、
    智谱、Ollama 等兼容接口只有 /v1/chat/completions，请继续使用 OpenAIAdapter。
    """

    def create_client(self) -> Any:
        """创建OpenAI客户端"""
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def create_async_client(self) -> Any:
        """创建OpenAI异步客户端"""
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    # ==================== 请求格式转换 ====================

    @classmethod
    def _convert_content(cls, content: Any) -> Any:
        """将 Chat Completions 风格的多模态内容块转换为 Responses 风格

        - {"type": "text", "text": ...}            -> {"type": "input_text", "text": ...}
        - {"type": "image_url", "image_url": {...}} -> {"type": "input_image", "image_url": "..."}
        """
        if not isinstance(content, list):
            return content

        converted = []
        for block in content:
            if not isinstance(block, dict):
                converted.append(block)
                continue

            block_type = block.get("type")
            if block_type == "text":
                converted.append({"type": "input_text", "text": block.get("text", "")})
            elif block_type == "image_url":
                image_url = block.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    detail = image_url.get("detail")
                else:
                    url, detail = image_url, None

                item = {"type": "input_image", "image_url": url}
                if detail:
                    item["detail"] = detail
                converted.append(item)
            else:
                converted.append(block)

        return converted

    def _convert_messages(self, messages: List[Dict]) -> tuple[Optional[str], List[Dict]]:
        """将统一消息格式转换为 Responses 的 (instructions, input) 结构"""
        instructions_parts = []
        input_items = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                if content:
                    instructions_parts.append(content)
            elif role == "tool":
                # 工具执行结果 -> function_call_output 条目
                if isinstance(content, str):
                    output = content
                else:
                    output = json.dumps(content, ensure_ascii=False)
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id"),
                    "output": output,
                })
            elif role == "assistant" and msg.get("responses_output"):
                # 原始条目已经包含文本与 function_call，不能再次重建以免重复。
                input_items.extend(msg["responses_output"])
            elif role == "assistant" and msg.get("tool_calls"):
                # 助手文本 + 工具调用 -> 文本消息条目 + 若干 function_call 条目
                if content:
                    input_items.append({
                        "role": "assistant",
                        "content": self._convert_content(content),
                    })
                for tool_call in msg["tool_calls"]:
                    function = tool_call.get("function", {})
                    arguments = function.get("arguments", "")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    input_items.append({
                        "type": "function_call",
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": arguments,
                    })
            else:
                input_items.append({
                    "role": role,
                    "content": self._convert_content(content) if content is not None else "",
                })

        instructions = "\n\n".join(instructions_parts) if instructions_parts else None
        return instructions, input_items

    @staticmethod
    def _convert_tools(tools: List[Dict]) -> List[Dict]:
        """将统一（Chat Completions）工具 schema 转换为 Responses 扁平 schema"""
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                function = tool["function"]
                entry = {
                    "type": "function",
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters") or {
                        "type": "object",
                        "properties": {},
                    },
                }
                if function.get("strict") is not None:
                    entry["strict"] = function["strict"]
                converted.append(entry)
            else:
                converted.append(tool)
        return converted

    @staticmethod
    def _convert_tool_choice(tool_choice: Any) -> Any:
        """将 Chat Completions 风格的 tool_choice 转换为 Responses 风格"""
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if function and function.get("name"):
                return {"type": "function", "name": function["name"]}
            if tool_choice.get("name"):
                return tool_choice
        return tool_choice

    @staticmethod
    def _prune_none(params: Dict) -> Dict:
        """剔除值为 None 的参数，避免向接口发送显式 null"""
        return {key: value for key, value in params.items() if value is not None}

    def _build_request(self, messages: List[Dict], **kwargs) -> Dict:
        """构建 responses.create 的请求参数"""
        instructions, input_items = self._convert_messages(messages)

        # max_tokens 与 max_output_tokens 互认，统一落到 max_output_tokens
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = kwargs.pop("max_output_tokens", None)
        else:
            kwargs.pop("max_output_tokens", None)

        request = {"model": self.model, "input": input_items, **self._prune_none(kwargs)}
        if instructions:
            request["instructions"] = instructions
        if max_tokens is not None:
            request["max_output_tokens"] = max_tokens

        return request

    # ==================== 响应解析 ====================

    @staticmethod
    def _check_response_status(response: Any, status: Optional[str] = None) -> None:
        """失败或不完整的输出不可作为成功结果交给上层。"""
        status = status or getattr(response, "status", None)
        if status is None or status == "completed":
            return
        error = getattr(response, "error", None)
        details = getattr(response, "incomplete_details", None)
        reason = (
            getattr(error, "message", None)
            or getattr(details, "reason", None)
            or getattr(error, "code", None)
            or status
        )
        raise TensorInpaintingException(f"Responses 状态 {status}: {reason}")

    @classmethod
    def _check_stream_event(cls, event: Any) -> None:
        event_type = getattr(event, "type", None)
        if event_type in ("response.failed", "response.incomplete", "response.cancelled"):
            cls._check_response_status(getattr(event, "response", None), event_type.split(".")[1])
        elif event_type == "error":
            raise TensorInpaintingException(
                f"Responses 流错误: {getattr(event, 'message', None) or getattr(event, 'code', 'unknown')}"
            )

    @staticmethod
    def _serialize_output(response: Any) -> List[Dict]:
        """保留所有原始字段，包括 reasoning、加密状态和 assistant phase。"""
        def serialize(item):
            if hasattr(item, "model_dump"):
                return item.model_dump(mode="json", exclude_none=True)
            return vars(item)

        return json.loads(json.dumps(getattr(response, "output", None) or [], default=serialize))

    @staticmethod
    def _extract_usage(usage: Any) -> Dict[str, int]:
        """Responses 的 usage 字段名与 Chat Completions 不同，统一映射"""
        if not usage:
            return {}
        return {
            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    @staticmethod
    def _item_text(item: Any) -> Optional[str]:
        """兼容对象/字典两种形态的取值"""
        if isinstance(item, dict):
            return item.get("text")
        return getattr(item, "text", None)

    @classmethod
    def _extract_output(cls, response: Any) -> tuple[str, List[ToolCall], Optional[str]]:
        """从 Responses 结果中解析文本、工具调用与推理摘要"""
        content_parts = []
        tool_calls = []
        reasoning_parts = []

        for item in getattr(response, "output", None) or []:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for part in getattr(item, "content", None) or []:
                    if getattr(part, "type", None) == "output_text":
                        text = cls._item_text(part)
                        if text:
                            content_parts.append(text)
            elif item_type == "function_call":
                tool_calls.append(ToolCall(
                    id=getattr(item, "call_id", None) or getattr(item, "id", "") or "",
                    name=getattr(item, "name", "") or "",
                    arguments=getattr(item, "arguments", "") or "",
                ))
            elif item_type == "reasoning":
                for summary in getattr(item, "summary", None) or []:
                    text = cls._item_text(summary)
                    if text:
                        reasoning_parts.append(text)

        content = "".join(content_parts)
        if not content:
            content = getattr(response, "output_text", None) or ""

        reasoning_content = "\n".join(reasoning_parts) if reasoning_parts else None
        return content, tool_calls, reasoning_content

    # ==================== 调用实现 ====================

    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()

        try:
            response = self._client.responses.create(**self._build_request(messages, **kwargs))
            self._check_response_status(response)
            latency_ms = int((time.time() - start_time) * 1000)

            content, _, reasoning_content = self._extract_output(response)

            return LLMResponse(
                content=content,
                model=getattr(response, "model", None) or self.model,
                usage=self._extract_usage(getattr(response, "usage", None)),
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )

        except Exception as e:
            raise TensorInpaintingException(f"OpenAI Responses API调用失败: {str(e)}")

    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        self.last_stats = None
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()

        stream = None
        try:
            stream = self._client.responses.create(**self._build_request(messages, stream=True, **kwargs))

            reasoning_parts = []
            usage = {}
            finished = False

            for event in stream:
                self._check_stream_event(event)
                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta
                elif event_type == "response.reasoning_summary_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        reasoning_parts.append(delta)
                elif event_type == "response.completed":
                    completed = getattr(event, "response", None)
                    self._check_response_status(completed)
                    finished = True
                    if completed is not None:
                        usage = self._extract_usage(getattr(completed, "usage", None))

            if not finished:
                raise TensorInpaintingException("Responses 流提前结束，未收到 response.completed")
            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content="".join(reasoning_parts) if reasoning_parts else None
            )

        except Exception as e:
            raise TensorInpaintingException(f"OpenAI Responses API流式调用失败: {str(e)}")
        finally:
            if stream is not None and hasattr(stream, "close"):
                stream.close()

    async def astream_invoke(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        """真正的异步流式调用（使用 OpenAI 原生异步客户端）"""
        self.last_stats = None
        if not self._async_client:
            self._async_client = self.create_async_client()

        start_time = time.time()

        stream = None
        try:
            stream = await self._async_client.responses.create(
                **self._build_request(messages, stream=True, **kwargs)
            )

            reasoning_parts = []
            usage = {}
            finished = False

            async for event in stream:
                self._check_stream_event(event)
                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta
                elif event_type == "response.reasoning_summary_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        reasoning_parts.append(delta)
                elif event_type == "response.completed":
                    completed = getattr(event, "response", None)
                    self._check_response_status(completed)
                    finished = True
                    if completed is not None:
                        usage = self._extract_usage(getattr(completed, "usage", None))

            if not finished:
                raise TensorInpaintingException("Responses 流提前结束，未收到 response.completed")
            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content="".join(reasoning_parts) if reasoning_parts else None
            )

        except Exception as e:
            raise TensorInpaintingException(f"OpenAI Responses API异步流式调用失败: {str(e)}")
        finally:
            if stream is not None and hasattr(stream, "close"):
                await stream.close()

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict],
                         tool_choice: Union[str, Dict] = "auto", **kwargs) -> LLMToolResponse:
        """工具调用（Function Calling）"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        try:
            request = self._build_request(messages, **kwargs)
            # store=False 时也能通过加密 reasoning 条目续接工具调用。
            request["include"] = list(request.get("include") or [])
            if "reasoning.encrypted_content" not in request["include"]:
                request["include"].append("reasoning.encrypted_content")
            if tools:
                request["tools"] = self._convert_tools(tools)
                if tool_choice is not None:
                    request["tool_choice"] = self._convert_tool_choice(tool_choice)

            response = self._client.responses.create(**request)
            self._check_response_status(response)
            latency_ms = int((time.time() - start_time) * 1000)

            content, tool_calls, _ = self._extract_output(response)

            return LLMToolResponse(
                content=content if content else None,
                tool_calls=tool_calls,
                model=getattr(response, "model", None) or self.model,
                usage=self._extract_usage(getattr(response, "usage", None)),
                latency_ms=latency_ms,
                responses_output=self._serialize_output(response),
            )

        except Exception as e:
            raise TensorInpaintingException(f"OpenAI Responses Function Calling调用失败: {str(e)}")


class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude适配器

    处理Claude特有的消息格式：
    - system参数独立（不在messages中）
    - 消息格式转换
    """

    def create_client(self) -> Any:
        """创建Anthropic客户端"""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise TensorInpaintingException(
                "使用Anthropic需要安装: pip install anthropic"
            )

        return Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _convert_messages(self, messages: List[Dict]) -> tuple[Optional[str], List[Dict]]:
        """转换消息格式，提取system消息"""
        system_content = None
        converted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})

                for tool_call in msg["tool_calls"]:
                    function = tool_call.get("function", {})
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                    content_blocks.append({
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": function.get("name"),
                        "input": arguments,
                    })

                converted_messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
            elif msg["role"] == "tool":
                converted_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": msg.get("content", ""),
                    }],
                })
            else:
                converted_messages.append(msg)

        return system_content, converted_messages

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """将统一的 OpenAI 风格工具 schema 转换为 Anthropic 工具 schema"""
        converted_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                function = tool["function"]
                converted_tools.append({
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {
                        "type": "object",
                        "properties": {},
                    }),
                })
            else:
                converted_tools.append(tool)

        return converted_tools

    def _convert_tool_choice(self, tool_choice: Any) -> Optional[Dict]:
        """将统一的 tool_choice 转换为 Anthropic 格式"""
        if tool_choice in (None, "auto", "none"):
            return None
        if tool_choice == "required":
            return {"type": "any"}
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if function and function.get("name"):
                return {"type": "tool", "name": function["name"]}
        return tool_choice

    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_content, converted_messages = self._convert_messages(messages)

        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs
            }
            if system_content:
                request_params["system"] = system_content

            response = self._client.messages.create(**request_params)

            latency_ms = int((time.time() - start_time) * 1000)

            # 提取内容
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text

            # 提取usage
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise TensorInpaintingException(f"Anthropic API调用失败: {str(e)}")

    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_content, converted_messages = self._convert_messages(messages)

        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                "stream": True,
                **kwargs
            }
            if system_content:
                request_params["system"] = system_content

            usage = {}

            with self._client.messages.stream(**request_params) as stream:
                for text in stream.text_stream:
                    yield text

                # 获取最终消息以提取usage
                final_message = stream.get_final_message()
                if hasattr(final_message, 'usage') and final_message.usage:
                    usage = {
                        "prompt_tokens": final_message.usage.input_tokens,
                        "completion_tokens": final_message.usage.output_tokens,
                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise TensorInpaintingException(f"Anthropic API流式调用失败: {str(e)}")

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> LLMToolResponse:
        """工具调用（Anthropic格式）"""
        if not self._client:
            self._client = self.create_client()

        system_content, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)
        tool_choice = self._convert_tool_choice(kwargs.pop("tool_choice", None))

        start_time = time.time()
        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "tools": converted_tools,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs
            }
            if system_content:
                request_params["system"] = system_content
            if tool_choice:
                request_params["tool_choice"] = tool_choice

            response = self._client.messages.create(**request_params)
            latency_ms = int((time.time() - start_time) * 1000)

            content = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input)
                    ))

            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }

            return LLMToolResponse(
                content=content if content else None,
                tool_calls=tool_calls,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise TensorInpaintingException(f"Anthropic工具调用失败: {str(e)}")


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini适配器

    处理Gemini特有的API格式
    使用新版 google.genai 包（替代已废弃的 google.generativeai）
    """

    def create_client(self) -> Any:
        """创建Gemini客户端"""
        try:
            from google import genai
        except ImportError:
            raise TensorInpaintingException(
                "使用Gemini需要安装: pip install google-genai"
            )

        client = genai.Client(api_key=self.api_key)
        return client

    def _convert_messages(self, messages: List[Dict]) -> tuple[Optional[str], List[Dict]]:
        """转换消息格式"""
        from google.genai import types as genai_types

        system_instruction = None
        converted_messages = []
        tool_call_names = {}

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                parts = []
                if msg.get("content"):
                    parts.append(genai_types.Part.from_text(text=msg["content"]))

                for tool_call in msg["tool_calls"]:
                    function = tool_call.get("function", {})
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                    tool_name = function.get("name", "")
                    tool_call_id = tool_call.get("id")
                    if tool_call_id and tool_name:
                        tool_call_names[tool_call_id] = tool_name

                    parts.append(genai_types.Part.from_function_call(
                        name=tool_name,
                        args=arguments,
                    ))

                converted_messages.append(genai_types.Content(
                    role="model",
                    parts=parts,
                ))
            elif msg["role"] == "tool":
                tool_name = tool_call_names.get(msg.get("tool_call_id"), "tool_result")
                converted_messages.append(genai_types.Content(
                    role="tool",
                    parts=[genai_types.Part.from_function_response(
                        name=tool_name,
                        response={"result": msg.get("content", "")},
                    )],
                ))
            else:
                # Gemini使用 "user" 和 "model" 作为角色
                role = "model" if msg["role"] == "assistant" else "user"
                converted_messages.append(genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=msg["content"] or "")],
                ))

        return system_instruction, converted_messages

    def _convert_tool_choice(self, tool_choice: Any) -> Optional[Any]:
        """将统一的 tool_choice 转换为 Gemini 工具配置"""
        from google.genai import types as genai_types

        if tool_choice in (None, "auto"):
            return None
        if tool_choice == "none":
            return genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(mode="NONE")
            )
        if tool_choice == "required":
            return genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(mode="ANY")
            )
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if function and function.get("name"):
                return genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[function["name"]],
                    )
                )
        return tool_choice

    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        from google.genai import types as genai_types

        start_time = time.time()
        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            # 创建生成配置
            config_params = {}
            if "temperature" in kwargs:
                config_params["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                config_params["max_output_tokens"] = kwargs.pop("max_tokens")
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            response = self._client.models.generate_content(
                model=self.model,
                contents=converted_messages,
                config=genai_types.GenerateContentConfig(**config_params) if config_params else None
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # 提取内容
            content = response.text if hasattr(response, 'text') else ""

            # 提取usage
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                    "total_tokens": response.usage_metadata.total_token_count or 0,
                }

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise TensorInpaintingException(f"Gemini API调用失败: {str(e)}")

    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        from google.genai import types as genai_types

        start_time = time.time()
        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            # 创建生成配置
            config_params = {}
            if "temperature" in kwargs:
                config_params["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                config_params["max_output_tokens"] = kwargs.pop("max_tokens")
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            usage = {}

            response = self._client.models.generate_content_stream(
                model=self.model,
                contents=converted_messages,
                config=genai_types.GenerateContentConfig(**config_params) if config_params else None
            )

            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text

                # 尝试提取usage（可能在最后一个chunk）
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage = {
                        "prompt_tokens": chunk.usage_metadata.prompt_token_count or 0,
                        "completion_tokens": chunk.usage_metadata.candidates_token_count or 0,
                        "total_tokens": chunk.usage_metadata.total_token_count or 0,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise TensorInpaintingException(f"Gemini API流式调用失败: {str(e)}")

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> LLMToolResponse:
        """工具调用（Gemini格式）"""
        if not self._client:
            self._client = self.create_client()

        from google.genai import types as genai_types

        system_instruction, converted_messages = self._convert_messages(messages)
        tool_choice = self._convert_tool_choice(kwargs.pop("tool_choice", None))

        start_time = time.time()
        try:
            # 转换工具格式为Gemini格式
            gemini_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    gemini_tools.append(
                        genai_types.FunctionDeclaration(
                            name=func["name"],
                            description=func.get("description", ""),
                            parameters_json_schema=func.get("parameters", {})
                        )
                    )

            config_params = {}
            if "temperature" in kwargs:
                config_params["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                config_params["max_output_tokens"] = kwargs.pop("max_tokens")
            if gemini_tools:
                config_params["tools"] = [genai_types.Tool(function_declarations=gemini_tools)]
            if system_instruction:
                config_params["system_instruction"] = system_instruction
            if tool_choice:
                config_params["tool_config"] = tool_choice

            response = self._client.models.generate_content(
                model=self.model,
                contents=converted_messages,
                config=genai_types.GenerateContentConfig(**config_params) if config_params else None
            )
            latency_ms = int((time.time() - start_time) * 1000)

            content = response.text if hasattr(response, 'text') else ""
            tool_calls = []

            # 解析 Gemini 工具调用
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        tool_calls.append(ToolCall(
                            id=f"call_{int(time.time()*1000)}",  # Gemini 没有显式的 call_id，生成一个
                            name=part.function_call.name,
                            arguments=json.dumps(dict(part.function_call.args))
                        ))

            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                    "total_tokens": response.usage_metadata.total_token_count or 0
                }

            return LLMToolResponse(
                content=content if content else None,
                tool_calls=tool_calls,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise TensorInpaintingException(f"Gemini工具调用失败: {str(e)}")


def create_adapter(
    api_key: str,
    base_url: Optional[str],
    timeout: int,
    model: str,
    api_style: Optional[str] = None
) -> BaseLLMAdapter:
    """
    根据base_url自动选择适配器

    检测逻辑：
    - anthropic.com -> AnthropicAdapter
    - googleapis.com 或 generativelanguage -> GeminiAdapter
    - api_style=responses -> OpenAIResponsesAdapter（OpenAI 官方 Responses API）
    - 其他 -> OpenAIAdapter（默认，兼容所有OpenAI格式接口）

    Args:
        api_style: OpenAI 接口风格，"chat"（默认）或 "responses"；
            为 None 时读取 LLM_API_STYLE 环境变量
    """
    if base_url:
        base_url_lower = base_url.lower()

        if "anthropic.com" in base_url_lower:
            return AnthropicAdapter(api_key, base_url, timeout, model)

        if "googleapis.com" in base_url_lower or "generativelanguage" in base_url_lower:
            return GeminiAdapter(api_key, base_url, timeout, model)

    style = (api_style or os.getenv("LLM_API_STYLE") or "chat").strip().lower()
    if style in ("responses", "response"):
        return OpenAIResponsesAdapter(api_key, base_url, timeout, model)

    # 默认使用OpenAI适配器（兼容所有OpenAI格式接口）
    return OpenAIAdapter(api_key, base_url, timeout, model)
