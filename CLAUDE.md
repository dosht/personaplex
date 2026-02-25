# personaplex

## Project Overview

<!-- Describe your project: what it does, who it's for, and key features -->

This project uses the Claude Team agentic system for deterministic, workflow-driven development.

## Tech Stack

<!-- Add your tech stack here -->

## Agent Commands

Activate specialized agents for different tasks:

- `/team:developer` - Implement features from user stories
- `/team:tech-lead` - Review code and enforce standards
- `/team:architect` - Review architecture and design decisions
- `/team:manual-tester` - Test features and validate acceptance criteria
- `/team:scrum-planner` - Plan sprints with upfront story refinement (Scrum)
- `/team:kanban-planner` - Just-in-time story creation with pull-based workflow (Kanban)
- `/team:react-ui-designer` - Design beautiful, accessible React UIs
- `/team:mentor` - Train junior developers with adaptive tutorials

## Cross-Project Tags

This project uses the following knowledge tags for cross-project search:

No tags defined yet

## Development Workflow

1. **Story Creation**: Product Manager creates stories in `docs/product/epics/`
2. **Implementation**: Developer implements features following stories
3. **Testing**: Manual Tester validates acceptance criteria
4. **Review**: Tech Lead and Architect review implementation
5. **Done**: Story marked as done when all criteria met

## Story Management

Stories are managed through the unified MCP server. All agents have access to:

- `mcp__claude-team__list_stories` - List stories with filters
- `mcp__claude-team__get_story` - Get full story details
- `mcp__claude-team__update_status` - Update story status
- `mcp__claude-team__create_story` - Create new stories
- `mcp__claude-team__get_epic_progress` - View epic progress

## Knowledge Recording

Agents automatically record knowledge to help future projects learn from decisions, lessons, and patterns.

### Automatic Recording

Different agents record knowledge based on their activities:

- **Architect**: After creating an ADR, automatically records it in the knowledge base
- **Tech Lead**: After researching best practices, offers to record findings as lessons or patterns
- **Developer**: After solving a tricky bug, suggests recording the solution as a lesson

### Manual Recording

You can ask any agent to record knowledge using these phrases:

- "Add this to the knowledge base"
- "Record this as a lesson learned"
- "Save this pattern for future projects"
- "Document this decision for other projects"

The agent will create the appropriate knowledge entry and confirm with tags.

### What to Record

**Lessons** - Things that worked/didn't work, gotchas, tips
- Example: "Always validate JWT expiry at 80% lifetime, not 100%"
- Tags: authentication, jwt, security

**Decisions** - Architectural choices with context and tradeoffs
- Example: "Use React Query over Redux for server state management"
- Tags: react, state-management, architecture

**Patterns** - Reusable code patterns and best practices
- Example: "Form validation using Zod schema + react-hook-form"
- Tags: react, forms, validation, typescript

### Recording Format

When an agent records knowledge, it:

1. Creates a full markdown file in `docs/knowledge/` or `docs/architecture/decisions/`
2. Calls the appropriate MCP tool (`record_lesson`, `record_decision`, or `record_pattern`)
3. Confirms: "Recorded [type] '[name]' with tags [tag1, tag2, tag3]"

The knowledge is then indexed globally and searchable across all your projects.

### Quality Guidelines

Recorded knowledge should include:

- **Concise summary** - One sentence that captures the essence
- **Relevant tags** - 3-5 tags that help discovery
- **Context** - Why this matters, when it applies, what problem it solves
- **Examples** - Code snippets or concrete examples where applicable
- **Tradeoffs** - What are the downsides or alternatives considered

## Project Structure

```
.
├── .claude-team.yaml          # Claude Team project configuration
├── CLAUDE.md                  # This file
├── docs/
│   ├── product/
│   │   ├── README.md          # Product overview and progress
│   │   └── epics/             # Epic directories with stories
│   └── knowledge/
│       ├── lessons/           # Lessons learned
│       └── patterns/          # Reusable patterns
└── ...                        # Your project files
```

## Getting Started

1. Register this project: `claude-team init` (already done)
2. Create your first epic: Create directory in `docs/product/epics/`
3. Create stories: Use `mcp__claude-team__create_story` or create manually
4. Start developing: Activate `/team:developer` and implement stories
