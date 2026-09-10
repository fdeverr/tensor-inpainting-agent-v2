from pathlib import Path

from research_agent.agent_tools import framework


def test_framework_uses_the_sibling_tensor_inpainting_agent_checkout():
    expected = Path(framework.__file__).resolve().parents[2] / "__init__.py"
    actual = Path(framework.tensor_inpainting_agent.__file__).resolve()
    assert actual == expected
