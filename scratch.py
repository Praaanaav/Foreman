import orchestrator.travel_tools  # importing this registers the tools
from orchestrator.registry import registry
from orchestrator.schemas import AgentType

r = registry.call("get_weather", AgentType.RESEARCH, city="Lisbon", days=3)
print(r.success, r.latency_ms, "ms")
print(r.output or r.error)

r = registry.call("write_file", AgentType.WRITER, filename="test.md", content="hello")
print(r.output)

try:
    registry.call("write_file", AgentType.RESEARCH, filename="x.md", content="no")
except PermissionError as e:
    print("Blocked:", e)

print(len(registry.call_log), "calls logged")