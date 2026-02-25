# Product Documentation

This directory contains all product-related documentation including epics and user stories.

## Directory Structure

- `epics/` - Contains epic directories with stories and epic-level documentation
- Each epic directory follows the pattern: `{EPIC-ID}-{epic-name}/`
- Stories within epics are numbered sequentially: `{EPIC-ID}-{NN}-{description}.md`

## Using This Documentation

Product managers, developers, and stakeholders can use this documentation to:
- Understand product roadmap and priorities
- Track progress on features and epics
- Access detailed specifications for implementation
- Review acceptance criteria and testing requirements

## Project Progress

<!-- PROGRESS_START -->
**Overall Progress:** [░░░░░░░░░░░░░░░░░░░░] 0% (0/0 stories done)
*Last updated: Not yet calculated*

### Epic Progress

No epics yet. Create your first epic directory to get started.

<!-- PROGRESS_END -->

## Creating a New Epic

1. Create a directory under `epics/` with the epic ID (e.g., `PROJ-01-feature-name`)
2. Add a `README.md` describing the epic
3. Add story files with the story ID (e.g., `PROJ-01-01-first-story.md`)

## Story Format

Stories use markdown with frontmatter metadata:

```markdown
---
story_id: PROJ-01-01
epic_id: PROJ-01
title: Story Title
status: draft
priority: medium
points: 3
---
# PROJ-01-01: Story Title

## User Story
**As a** [user type]
**I want** [goal]
**So that** [benefit]

## Acceptance Criteria
...
```
