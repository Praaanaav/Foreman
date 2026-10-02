"""Travel-domain data models shared across the orchestration system."""
from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    DESTINATION = "destination"
    TRANSPORTATION = "transportation"
    ACCOMMODATION = "accommodation"
    BUDGET = "budget"
    ITINERARY = "itinerary"
    TRAVEL_INFO = "travel_info"
    REVIEWER = "reviewer"


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


class TripRequirements(BaseModel):
    origin: str | None = Field(
        default=None, description="Where the travelers start from"
    )
    destinations: list[str] = Field(
        description="Countries/regions the trip covers"
    )
    days: int = Field(description="Total trip length in days")
    travelers: int = Field(description="Number of travelers")
    budget_total: float | None = Field(
        default=None, description="Total budget in `currency`"
    )
    currency: str = "INR"
    food_preferences: list[str] = Field(default_factory=list)
    pace: str = Field(default="moderate", description="slow / moderate / fast")
    interests: list[str] = Field(default_factory=list)
    accommodation: str | None = None
    constraints: list[str] = Field(default_factory=list)


class TravelPlan(BaseModel):
    """Supervisor's task decomposition."""

    goal: str
    requirements: TripRequirements
    subtasks: list[SubTask]
    confidence: float = 0.9


class AgentOutput(BaseModel):
    """Standard envelope returned by every specialist agent."""

    content: str = Field(description="The research/plan result as readable text")
    confidence: float = 0.8
    sources: list[str] = Field(default_factory=list)
    needs_attention: bool = Field(
        default=False,
        description="True when the agent cannot confidently verify something",
    )
    attention_reason: str | None = None


class CostItem(BaseModel):
    category: str
    amount: float
    currency: str = "INR"
    details: str = ""


class BudgetReport(BaseModel):
    items: list[CostItem]
    total: float
    currency: str
    budget_limit: float | None
    within_budget: bool
    notes: str = ""


class Activity(BaseModel):
    name: str
    time_slot: str = Field(description="morning / afternoon / evening / all-day")
    estimated_cost: float | None = None
    notes: str | None = None


class ItineraryDay(BaseModel):
    day: int
    city: str
    theme: str | None = None
    activities: list[Activity]
    travel: str | None = Field(
        default=None, description="Travel to next base, if any, on this day"
    )


class ItineraryStop(BaseModel):
    city: str
    nights: int
    check_in_day: int
    check_out_day: int
    hotel: str | None = None


class Itinerary(BaseModel):
    title: str
    bases: list[ItineraryStop]
    days: list[ItineraryDay]
    summary: str


class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


class ReviewResult(BaseModel):
    verdict: ReviewVerdict
    issues: list[str] = Field(default_factory=list)
    feedback: str = ""
    confidence: float
    escalation_reason: str | None = None


class HumanAction(str, Enum):
    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"
    TAKE_OVER = "take_over"


class HumanDecision(BaseModel):
    action: HumanAction
    modifications: str | None = Field(
        default=None,
        description="Free text: requested changes, or the full itinerary when taking over",
    )


class TraceEvent(BaseModel):
    agent: str
    event: str = Field(
        description="llm_call / tool_call / decision / retry / escalation / human / memory"
    )
    detail: dict = Field(default_factory=dict)
    latency_ms: float | None = None
    timestamp: str


class TravelState(TypedDict, total=False):
    """LangGraph shared state for one trip-planning run.

    The state doubles as short-term trip memory for the duration of the run.
    """

    trip_id: str
    traveler_id: str
    request: str
    profile: dict
    memories: list[str]
    plan: dict
    errors: Annotated[list[str], operator.add]
    destination_research: dict
    transport_research: dict
    accommodation_research: dict
    travel_info: dict
    budget_report: dict
    itinerary: dict
    review: dict
    attempts: int
    escalation: dict | None
    human_decision: dict | None
    final_itinerary: str
    trace: Annotated[list[dict], operator.add]
