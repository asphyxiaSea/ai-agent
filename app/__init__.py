"""Application package for the ai-agent service.

Layered structure:
- core: settings and shared config
- domain: schema definitions
- infra: external clients
- workflows: LangGraph graphs, nodes and state
- application: pipeline orchestration entrypoints
- api: FastAPI routers and request models
"""
