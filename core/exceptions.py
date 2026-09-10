"""异常体系"""

class TensorInpaintingException(Exception):
    """TensorInpainting基础异常类"""
    pass

class LLMException(TensorInpaintingException):
    """LLM相关异常"""
    pass

class AgentException(TensorInpaintingException):
    """Agent相关异常"""
    pass

class ConfigException(TensorInpaintingException):
    """配置相关异常"""
    pass

class ToolException(TensorInpaintingException):
    """工具相关异常"""
    pass
