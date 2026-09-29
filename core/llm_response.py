"""LLM响应对象定义"""

from typing import Optional, Dict, List
from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """统一的工具调用对象"""
    id: str
    name: str
    arguments: str


@dataclass
class LLMToolResponse:
    """统一的工具调用响应对象"""
    content: Optional[str]
    tool_calls: List[ToolCall]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    responses_output: List[Dict] = field(default_factory=list)
    """Responses 原始输出条目，供工具结果后的下一轮完整回传。"""

    def to_assistant_message(self) -> Dict:
        """构建工具调用历史，同时保留供应商所需的续接状态。"""
        message = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ],
        }
        if self.responses_output:
            message["responses_output"] = deepcopy(self.responses_output)
        return message


@dataclass
class LLMResponse:
    """
    统一的LLM响应对象
    
    包含响应内容、推理过程（thinking model）、token使用统计、耗时等信息
    """
    
    content: str
    """回复内容"""
    
    model: str
    """实际使用的模型名称"""
    
    usage: Dict[str, int] = field(default_factory=dict)
    """Token使用统计: {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}"""
    
    latency_ms: int = 0
    """调用耗时（毫秒）"""
    
    reasoning_content: Optional[str] = None
    """推理过程（仅thinking model如o1、deepseek-reasoner有此字段）"""
    
    def __str__(self) -> str:
        """向后兼容：直接打印返回content"""
        return self.content
    
    def __repr__(self) -> str:
        """详细信息展示"""
        parts = [
            f"LLMResponse(model={self.model}",
            f"latency={self.latency_ms}ms",
            f"tokens={self.usage.get('total_tokens', 0)}",
        ]
        if self.reasoning_content:
            parts.append("has_reasoning=True")
        parts.append(f"content_length={len(self.content)})")
        return ", ".join(parts)
    
    def to_dict(self) -> Dict:
        """转换为字典格式，方便日志记录"""
        result = {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        return result


@dataclass
class StreamStats:
    """
    流式调用的统计信息
    
    在流式调用结束后可通过 llm.last_call_stats 获取
    """
    
    model: str
    """实际使用的模型名称"""
    
    usage: Dict[str, int] = field(default_factory=dict)
    """Token使用统计"""
    
    latency_ms: int = 0
    """调用耗时（毫秒）"""
    
    reasoning_content: Optional[str] = None
    """推理过程（仅thinking model）"""
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {
            "model": self.model,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        return result
