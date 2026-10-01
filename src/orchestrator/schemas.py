from enum import Enum
from pydantic import BaseModel, Field


class AgentType(str, Enum):
    RESEARCH = "research"
    PLANNER = "planner"
    WRITER = "writer"


class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class SubTask(BaseModel):
    id: str = Field(description="Short unique id, e.g. 't1'")
    description: str = Field(description="What this subtask should accomplish")
    agent: AgentType = Field(description="Which specialist handles it")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of subtasks whose output this one needs",
    )
    required_inputs: list[str] = Field(
        default_factory=list,
        description="Information this subtask needs to start",
    )
    expected_output: str = Field(
        description="Format of the result, e.g. 'bullet list of 5 findings'"
    )
    complexity: Complexity = Complexity.MEDIUM
    status: Status = Status.PENDING
    result: str | None = None


class Plan(BaseModel):
    goal: str
    subtasks: list[SubTask]


class Task(BaseModel):
    user_id: str
    request: str