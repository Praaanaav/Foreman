# Travel Agent Orchestration System with Tool Use, Memory, and Human-in-the-Loop

## What You're Building

A domain-specific multi-agent travel orchestration platform where a Travel Supervisor Agent understands complex trip requests, decomposes them into travel-specific subtasks, delegates them to specialized travel agents, uses real-world travel tools and APIs, maintains persistent traveler preferences and trip memory, validates generated plans, and escalates decisions to a human when confidence is low or an action requires approval.

The system should be capable of turning a natural-language travel request into a personalized, validated, budget-aware, day-by-day travel plan.

---

# WHY THIS PROJECT LANDS INTERVIEWS

Travel is an excellent domain for demonstrating real-world agent engineering because planning a trip requires multiple interconnected decisions: destinations, transportation, accommodation, activities, budgets, schedules, and user preferences.

Instead of building another generic chatbot that generates an itinerary, this project demonstrates a multi-agent system that actually researches, compares, reasons, remembers, validates, and asks for human approval when necessary.

The architecture showcases the kind of autonomous AI workflow companies are building: specialized agents coordinating through a shared state, using external tools, learning from previous interactions, recovering from failures, and maintaining human oversight.

---

# TECH STACK

| COMPONENT | TOOL / LIBRARY | WHY THIS CHOICE |
|---|---|---|
| Language | Python 3.11+ | Ecosystem standard |
| Orchestration | LangGraph | Stateful multi-agent travel workflows |
| LLM Providers | OpenAI + Anthropic | Multi-model agent routing |
| Tool Framework | Custom + MCP | Extensible travel tool integration |
| Short-Term Memory | Redis | Fast task/session memory |
| Long-Term Memory | PostgreSQL + ChromaDB | Traveler profiles + semantic travel memory |
| Queue | Celery + Redis | Async travel-agent execution |
| Review UI | React or Streamlit | Human travel-plan approval |
| Containerization | Docker + docker-compose | Full system orchestration |
| Observability | OpenTelemetry | Agent/tool execution tracing |

---

# STEP-BY-STEP BUILD GUIDE

## Phase 1: Build the Travel Agent Architecture
**Day 1–4**

### 1. Design the Travel Agent Hierarchy

Create a travel-specific multi-agent architecture.

The **Travel Supervisor Agent** receives a complex travel request, understands the traveler's requirements, retrieves relevant memory, creates an execution plan, and delegates subtasks to specialized travel agents.

Specialist agents include:

- **Destination Research Agent** — researches destinations, attractions, activities, and local conditions.
- **Transportation Agent** — researches flights, trains, buses, transfers, and local transportation.
- **Accommodation Agent** — researches hotels and stays based on budget and preferences.
- **Budget Agent** — estimates and tracks total trip costs.
- **Itinerary Agent** — creates the day-by-day itinerary and checks travel time between activities.
- **Travel Information Agent** — researches weather, visa requirements, travel rules, events, and destination-specific information.

The **Travel Reviewer Agent** validates the outputs from these specialists before the final itinerary is returned.

Model each agent as a LangGraph node with defined input/output schemas.

---

### 2. Build the Travel Task Decomposition Engine

The supervisor's core capability is taking a request such as:

> "Plan a 10-day Japan trip for two people with a ₹1.5 lakh budget. We prefer vegetarian food, slower travel, and don't want to change hotels frequently."

and breaking it into ordered travel subtasks.

Example:

```text
1. Understand traveler preferences
2. Identify suitable Japanese destinations
3. Research transportation between destinations
4. Find suitable accommodations
5. Research activities and attractions
6. Estimate transportation + accommodation + activity costs
7. Construct day-by-day itinerary
8. Validate schedule and budget
9. Review final itinerary
10. Request human approval if required
```

Include dependencies.

Example:

```text
Destination Research
        ↓
Transportation Research
        ↓
Accommodation Research
        ↓
Activity Selection
        ↓
Budget Calculation
        ↓
Itinerary Generation
        ↓
Travel Review
```

Use structured output to enforce:

- Travel objective
- Destinations
- Subtasks
- Assigned specialist
- Dependencies
- Required inputs
- Expected output
- Estimated complexity
- Confidence

---

### 3. Build the Travel Tool Registry

Build a registry where travel tools are registered with:

- Name
- Description
- Input schema
- Output schema
- Agent permissions
- Rate limits
- Authentication requirements

Start with tools such as:

- Web search
- Destination search
- Flight search
- Hotel search
- Maps/distance calculation
- Weather lookup
- Currency conversion
- Travel-time calculation
- Attraction search
- Database query
- User preference lookup

Every tool invocation should be logged with:

```text
Agent
Tool
Inputs
Outputs
Latency
Cost
Success/Failure
Timestamp
```

Use MCP where appropriate so additional travel tools can be added without redesigning the agents.

---

### 4. Build the LangGraph Travel State Machine

Wire the travel agents into a LangGraph workflow:

```text
Travel Request
      ↓
Traveler Profile
      ↓
Memory Retrieval
      ↓
Travel Planning
      ↓
Parallel Travel Research
      ↓
Transportation / Hotels / Activities
      ↓
Budget Analysis
      ↓
Itinerary Generation
      ↓
Travel Review
      ↓
Synthesis
      ↓
Human Approval if Required
      ↓
Final Itinerary
```

Include conditional edges:

- Specialist fails → retry
- Travel information conflicts → research again
- Budget exceeds limit → replan
- Reviewer rejects itinerary → return with feedback
- Confidence is low → human escalation
- Sensitive booking action → human approval

---

# Phase 2: Build the Travel Memory System
**Day 4–7**

### 1. Implement Short-Term Trip Memory

During a travel-planning session, all agents share a working memory containing:

- Current trip requirements
- Destination candidates
- Selected destinations
- Transportation options
- Hotel options
- Activity options
- Budget calculations
- Itinerary drafts
- Intermediate research
- Tool results
- Errors
- Reviewer feedback

Store this in Redis for fast access.

This memory belongs to the current trip and can be cleared when the task is complete.

---

### 2. Build Long-Term Traveler Memory

After completing a trip-planning session, extract useful information about the traveler.

Example:

```text
Traveler Preferences
────────────────────────────
Budget: Moderate
Food: Vegetarian
Travel pace: Slow
Accommodation: Budget/Mid-range
Hotel changes: Prefer fewer
Activities: Culture + nature
Preferred trip length: 7–14 days
```

Also remember previous travel interactions:

```text
Previous Trip:
Thailand

User Feedback:
"Too many activities per day."

Lesson:
Reduce daily activity density.
```

Store semantic memories in ChromaDB and structured traveler information in PostgreSQL.

---

### 3. Implement Memory Retrieval for Travel Planning

When the Travel Supervisor creates a new plan, retrieve:

- Similar previous trips
- Previous itineraries
- Traveler preferences
- Previous feedback
- Successful approaches
- Rejected approaches
- Relevant destination knowledge

Example:

```text
NEW REQUEST
"Plan Japan for me."

        ↓

MEMORY RETRIEVAL

Previous Preference:
Slow travel

Previous Feedback:
Too many activities

Accommodation:
Prefer fewer hotel changes

Food:
Vegetarian

        ↓

TRAVEL PLANNING PROMPT
```

The retrieved memories should directly influence the generated travel plan.

---

### 4. Add Travel Memory Management

Implement:

- Memory importance scoring
- Memory consolidation
- Duplicate-memory merging
- Memory expiration
- Traveler memory dashboard
- Trip history
- Memory inspection
- Memory deletion

The dashboard should allow the user to see:

> "What does my travel assistant remember about me?"

and allow them to delete memories they don't want stored.

---

# Phase 3: Build the Human-in-the-Loop Travel System
**Day 7–10**

### 1. Define Travel-Specific Escalation Triggers

The system should escalate to a human when:

- The supervisor has low confidence in the itinerary
- Travel information conflicts between sources
- A specialist fails repeatedly
- The itinerary exceeds the user's budget
- Visa/entry information cannot be confidently verified
- The system is about to perform a booking or other consequential action
- The reviewer rejects the itinerary
- The user explicitly requests human review

---

### 2. Build the Travel Approval Queue

When escalation occurs:

```text
Agent detects issue
       ↓
Pause travel workflow
       ↓
Package trip context
       ↓
Add to review queue
       ↓
Notify human
       ↓
Human reviews
       ↓
Approve / Modify / Reject
       ↓
Resume travel workflow
```

The review package should contain:

- Original travel request
- Traveler preferences
- Current itinerary
- Research sources
- Agent decisions
- Budget
- Proposed action
- Confidence
- Reason for escalation

---

### 3. Implement Granular Travel Approval Levels

Define different levels:

**Notify**

> Agent continues but informs the user.

**Approve Action**

> Human approves a specific action.

**Approve Plan**

> Human reviews the entire itinerary before proceeding.

**Take Over**

> Human directly takes control of the travel workflow.

Example:

```text
Generate itinerary
        ↓
No approval required

Search hotels
        ↓
No approval required

Recommend hotel
        ↓
No approval required

Book hotel
        ↓
Human approval required
```

---

### 4. Build the Travel Review Interface

Create a UI showing:

- Traveler request
- Traveler preferences
- Destination research
- Transportation options
- Hotel options
- Activities
- Budget
- Day-by-day itinerary
- Agent reasoning/context
- Confidence
- Sources
- Relevant memories
- Previous similar trips
- Reviewer feedback

Provide:

```text
[ APPROVE ]

[ MODIFY ]

[ REJECT ]

[ TAKE OVER ]
```

Also include a chat panel so the human can ask questions such as:

> "Why did you choose Kyoto instead of Osaka?"

or:

> "Can you reduce the total cost by ₹20,000?"

---

# Phase 4: Build Travel Observability and Debugging
**Day 10–12**

### 1. Implement Full Travel Execution Tracing

Every trip-planning session should generate a trace containing:

```text
User Request
    ↓
Memory Retrieval
    ↓
Supervisor Planning
    ↓
Destination Research
    ↓
Flight Research
    ↓
Hotel Research
    ↓
Activity Research
    ↓
Budget Calculation
    ↓
Itinerary Generation
    ↓
Reviewer Evaluation
    ↓
Human Approval
    ↓
Final Itinerary
```

Track:

- Agent decisions
- Tool calls
- Research results
- Memory retrieval
- Latency
- Token usage
- Errors
- Retries
- Reviewer decisions
- Human interventions

Use OpenTelemetry spans with custom travel attributes.

---

### 2. Build the Travel Trace Explorer

Create a visual workflow showing:

```text
Supervisor
├── Memory Retrieval
├── Destination Agent
│   ├── Web Search
│   └── Attraction Search
├── Transportation Agent
│   ├── Flight Search
│   └── Train Search
├── Hotel Agent
│   └── Hotel Search
├── Budget Agent
├── Itinerary Agent
└── Reviewer
```

Each node should show:

- Agent
- Decision
- Tool used
- Latency
- Cost
- Status
- Errors

Clicking a node should reveal the complete execution context.

---

### 3. Add Travel Cost and Performance Tracking

For every trip, track:

- LLM tokens
- Model usage
- Number of travel-tool calls
- Execution time
- Research time
- Human review time
- Estimated API cost
- Number of retries
- Number of escalations

Aggregate this across trips.

Example:

```text
Average Trip Planning Cost
Average Planning Time
Most Used Travel Tools
Most Common Escalation Reason
Average Number of Agents Used
Average Number of Research Calls
```

---

### 4. Build the Travel Replay System

Allow developers to replay a previous travel-planning session.

Example:

```text
Original Trip
      ↓
Destination Agent
      ↓
Hotel Agent
      ↓
Budget Agent
      ↓
Reviewer
```

Change an input such as:

```text
Budget:
₹1,50,000
        ↓
₹1,00,000
```

and replay the workflow to see how the itinerary changes.

This becomes extremely useful for debugging agent behavior and testing improvements.

---

# Phase 5: Integration and End-to-End Testing
**Day 12–13**

### 1. Build a Compelling Travel Demo

Create one complex travel request that demonstrates the entire system.

Example:

> "Plan a 10-day Japan trip for two people with a ₹1.5 lakh budget. We prefer vegetarian food, slow travel, cultural experiences, and minimal hotel changes."

Demonstrate:

```text
Supervisor
    ↓
Memory Retrieval
    ↓
Trip Decomposition
    ↓
Destination Research
    ↓
Transport Research
    ↓
Hotel Research
    ↓
Activity Research
    ↓
Budget Calculation
    ↓
Itinerary Generation
    ↓
Reviewer Detects Problem
    ↓
Supervisor Replans
    ↓
Human Approves Final Itinerary
```

This single scenario demonstrates almost every important part of the architecture.

---

### 2. Containerize the Full System

Docker Compose should include:

```text
Travel Orchestration API
        │
        ├── LangGraph
        ├── Redis
        ├── PostgreSQL
        ├── ChromaDB
        ├── Celery Workers
        ├── Travel Tool Services
        ├── Trace Explorer
        └── Human Review UI
```

Include a demo script that automatically runs the complete travel scenario.

---

### 3. Write End-to-End Tests

Test that:

- Travel requests produce valid execution plans
- Specialists receive the correct subtasks
- Destination research works
- Transportation research works
- Hotel research works
- Budget calculations are correct
- Itinerary dependencies are respected
- Reviewer catches deliberately bad itineraries
- Memory improves repeated trip planning
- Budget overruns trigger replanning
- Human escalation happens at the correct point
- Failed agents can recover
- Workflow can resume after human approval

---

# Phase 6: Polish for Portfolio
**Day 13–14**

### 1. Record the Demo

Show the complete travel lifecycle:

```text
Complex Travel Request
        ↓
Traveler Memory Retrieval
        ↓
Supervisor Planning
        ↓
Multiple Travel Agents
        ↓
Real Tool Calls
        ↓
Budget Analysis
        ↓
Reviewer Validation
        ↓
Itinerary Correction
        ↓
Human Approval
        ↓
Final Personalized Itinerary
        ↓
Trace Explorer
```

Keep the demo under 5 minutes.

---

### 2. Write the Portfolio Narrative

Frame the project as:

> "I built a domain-specific multi-agent travel orchestration system where specialized AI agents research destinations, transportation, accommodation, activities, and budgets to collaboratively create personalized travel plans. The system maintains persistent traveler memory, validates agent outputs, automatically recovers from failures, and escalates consequential decisions to humans. Every agent decision and tool call is observable and replayable."

Lead with an architecture diagram showing:

```text
                 TRAVEL SUPERVISOR
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    DESTINATION      TRANSPORT      ACCOMMODATION
       AGENT            AGENT            AGENT
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  BUDGET AGENT
                         │
                         ▼
                 ITINERARY AGENT
                         │
                         ▼
                 TRAVEL REVIEWER
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
           APPROVE               ESCALATE
              │                     │
              ▼                     ▼
          FINAL PLAN             HUMAN
                                REVIEW
```

---

# PROJECT POSITIONING

Do not describe this primarily as:

> "A generic multi-agent orchestration framework."

Instead, describe it as:

> **"An AI Travel Planning & Orchestration System powered by specialized autonomous agents, persistent traveler memory, real-world travel tools, automated validation, and human-in-the-loop approval."**

Travel is the **product/domain**, while multi-agent orchestration is the **technology powering it**.