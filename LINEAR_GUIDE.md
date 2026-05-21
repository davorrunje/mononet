# Linear Issue Guide

This guide covers managing Linear issues from the command line using the `linear` CLI.

## Prerequisites

```bash
# Verify the CLI is available
linear --version

# Authenticate if needed
linear auth login
```

## Creating Issues

```bash
# Basic issue (interactive)
linear issue create --team SYN --title "Fix authentication bug"

# With full details (non-interactive, recommended)
linear issue create \
  --team SYN \
  --title "Add package & test scaffold" \
  --description-file /tmp/issue_body.md \
  --priority 2 \
  --label "Enhancement"

# Assign to yourself on creation
linear issue create --team SYN --title "My task" --assignee self

# Start immediately after creation
linear issue create --team SYN --title "My task" --start
```

### Description File (Recommended for Complex Issues)

Always use `--description-file` for markdown content — inline `--description` strips formatting:

```bash
cat > /tmp/issue_body.md << 'EOF'
## Scope
Brief description of what needs to be done.

## Changes
- **`path/to/file.py`** — what changes and why
- **`path/to/other.py`** — what changes and why

## Definition of Done
- [ ] First acceptance criterion
- [ ] Second acceptance criterion
EOF

linear issue create \
  --team SYN \
  --title "Descriptive issue title" \
  --description-file /tmp/issue_body.md
```

### Priority Values
| Value | Meaning |
|---|---|
| `1` | Urgent |
| `2` | High |
| `3` | Medium |
| `4` | Low |

## Viewing Issues

```bash
# View issue by ID (with comments)
linear issue view SYN-1234

# View without comments
linear issue view SYN-1234 --no-comments

# Output as JSON
linear issue view SYN-1234 --json

# Open in browser
linear issue view SYN-1234 --web

# Print issue ID inferred from current git branch
linear issue id
```

## Listing and Querying Issues

```bash
# List your assigned issues
linear issue list

# Query by team and state
linear issue query --team SYN --state started

# Search by keyword
linear issue query --team SYN --search "cookiecutter"

# Filter by label
linear issue query --team SYN --label "AI generated"

# Filter by multiple states
linear issue query --team SYN --state backlog --state unstarted

# All issues (no assignee filter)
linear issue query --team SYN --all-assignees

# Increase result limit
linear issue query --team SYN --limit 100

# Output as JSON for scripting
linear issue query --team SYN --json
```

### State Values
`triage`, `backlog`, `unstarted`, `started`, `completed`, `canceled`

## Updating Issues

```bash
# Update title
linear issue update SYN-1234 --title "New title"

# Update state
linear issue update SYN-1234 --state started

# Add a label
linear issue update SYN-1234 --label "AI generated"

# Change priority
linear issue update SYN-1234 --priority 1

# Assign to yourself
linear issue update SYN-1234 --assignee self

# Update description from file
linear issue update SYN-1234 --description-file /tmp/updated_body.md

# Change team
linear issue update SYN-1234 --team IMP2
```

## Starting Work on an Issue

`linear issue start` transitions the issue to "In Progress" and creates a git branch:

```bash
# Start issue from current branch (prompts to pick issue)
linear issue start

# Start a specific issue
linear issue start SYN-1234

# Start from a specific git ref
linear issue start SYN-1234 --from-ref main

# Use a custom branch name instead of the auto-generated one
linear issue start SYN-1234 --branch my-custom-branch
```

The auto-generated branch name follows the pattern `username/syn-1234-issue-title-slug`.

## Comments

```bash
# List comments on an issue
linear issue comment list SYN-1234

# Add a comment (interactive editor opens)
linear issue comment add SYN-1234

# Update an existing comment
linear issue comment update <commentId>

# Delete a comment
linear issue comment delete <commentId>
```

## Creating a Pull Request from an Issue

```bash
# Create PR with issue details pre-filled (uses current branch)
linear issue pull-request SYN-1234

# Create as draft
linear issue pull-request SYN-1234 --draft

# Specify a custom title (issue ID is prepended automatically)
linear issue pull-request SYN-1234 --title "Add scaffold files"

# Specify base branch
linear issue pull-request SYN-1234 --base main

# Open in browser after creating
linear issue pull-request SYN-1234 --web
```

## Labels

```bash
# List all labels for a team
linear label list --team SYN

# Add label when creating
linear issue create --team SYN --title "..." --label "AI generated"

# Add label to existing issue
linear issue update SYN-1234 --label "AI generated"
```

## Typical Issue Workflow

```bash
# 1. Create issue with description
cat > /tmp/issue.md << 'EOF'
## Scope
What needs to be done.

## Definition of Done
- [ ] Acceptance criterion
EOF

linear issue create \
  --team SYN \
  --title "Feature: add X to Y" \
  --description-file /tmp/issue.md \
  --label "AI generated"

# 2. Start working — transitions state and creates git branch
linear issue start SYN-1234

# 3. Do the work, commit, push...

# 4. Create PR from the issue
linear issue pull-request SYN-1234 --draft

# 5. Mark ready when done
gh pr ready <PR_NUMBER>
```

## Deleting Issues

```bash
linear issue delete SYN-1234
```

## Teams

```bash
# List all teams
linear team list
```

## Raw GraphQL

For operations not covered by CLI commands:

```bash
# Execute a raw GraphQL query
linear api 'query { viewer { name email } }'

# Print the full schema
linear schema
```
