"""gemma — the public SDK surface (v0.1, the embedding seam).

The curriculum builds the harness one primitive per chapter across ``model/``,
``harness/``, and ``ui/``. This package is the curated, versioned surface external
code builds on: ``import gemma`` gives you exactly the agent, its structured
result, tools and their registry, the permission policy, the provider and model
seam, the editable-config door and its schema, and provenance — and nothing else.

The internal module paths (``from harness.agent import Agent``) keep working, so
adopting this façade is opt-in. What is exported here is the contract that
``__version__`` promises stability for; the rest of the tree may move between
chapters. See dev-notes/adr/0001 (why this evolves on its own axis) and 0002
(mechanism here, domain in the consumer).
"""

from __future__ import annotations

from harness.agent import Agent, run_once
from harness.harness_config import CONFIG, HarnessConfig, config_schema, load_config
from harness.observability import Tracer
from harness.policy import DEFAULT_MUTATORS, Policy
from harness.provenance import provenance
from harness.result import RunResult, ToolCall
from harness.tools import Tool, ToolRegistry, default_tools
from model import Provider, chat, fake, load_env

__version__ = "0.3.0"

__all__ = [
    "CONFIG",
    "DEFAULT_MUTATORS",
    "Agent",
    "HarnessConfig",
    "Policy",
    "Provider",
    "RunResult",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "Tracer",
    "__version__",
    "chat",
    "config_schema",
    "default_tools",
    "fake",
    "load_config",
    "load_env",
    "provenance",
    "run_once",
]
