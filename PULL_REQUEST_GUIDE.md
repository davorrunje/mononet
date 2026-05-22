# Pull Request Guide

This guide covers the complete pull request workflow: creating PRs with GitHub CLI, managing them, and responding to reviews.

## Creating Pull Requests with GitHub CLI

### Recommended Workflow: Description File First

**Always create PR descriptions in separate markdown files first:**

```bash
# 1. Create a detailed PR description in a markdown file
cat > /tmp/pr_description.md << 'EOF'
# Your PR Title Here

Brief summary of what this PR accomplishes.

## Changes Made
- List the specific changes
- Include any new files or modified functionality
- Mention any breaking changes

## Testing
- Describe testing performed
- Include any manual verification steps

## Related Issues
- Fixes SYN-1234 (if applicable)
- Closes #45 (if applicable)
EOF

# 2. Create PR using the description file
gh pr create --title "Your PR title" --body-file /tmp/pr_description.md
```

### Basic PR Creation Commands
```bash
# Using description file (RECOMMENDED)
gh pr create --title "Feature: add new functionality" --body-file /tmp/pr_description.md

# Inline description (for simple PRs only)
gh pr create --title "Fix typo in README" --body "Simple typo fix"

# Interactive mode (prompts for title/body)
gh pr create
```

### PR Creation Best Practices

#### 1. Always Use Description Files
- **Create markdown files** for PR descriptions rather than inline text
- **Template your descriptions** with consistent sections (Changes, Testing, etc.)
- **Review and edit** descriptions before creating the PR
- **Store important PRs** in `/tmp/` or project docs for reference

#### 2. Description Content Guidelines
- **Use descriptive titles** that summarize the change
- **Include comprehensive descriptions** with:
  - What was changed and why
  - Testing performed
  - Breaking changes (if any)
  - Related issues or tickets
- **Reference Linear issues** using format: `Fixes SYN-1234`
- **Add checklists** for complex changes

#### 3. Description Template Example
```markdown
# PR Title: Brief Summary of Changes

## Overview
Brief explanation of what this PR accomplishes and why it's needed.

## Changes Made
- **File 1**: Description of changes
- **File 2**: Description of changes
- **New Feature**: What was added

## Testing
- [ ] All tests pass locally
- [ ] Manual testing completed
- [ ] No regressions detected

## Related Issues
- Fixes SYN-1234
- Related to #45

## Additional Notes
Any other context or considerations for reviewers.
```

## Reading PR Comments and Reviews

### Accessing Feedback
```bash
# View PR with comments in terminal
gh pr view <PR_NUMBER> --comments

# View specific PR from different repo
gh pr view <PR_NUMBER> --repo owner/repo --comments

# Get comments as JSON for programmatic processing
gh api repos/owner/repo/pulls/<PR_NUMBER>/comments > /tmp/pr_comments.json

# Get general issue comments (not line-specific)
gh api repos/owner/repo/issues/<PR_NUMBER>/comments > /tmp/issue_comments.json

# Get reviews (approve/request changes/comment)
gh api repos/owner/repo/pulls/<PR_NUMBER>/reviews > /tmp/pr_reviews.json
```

### Understanding Comment Types
1. **General comments** - Overall feedback on the PR
2. **Line comments** - Specific feedback on code lines
3. **Review comments** - Formal review with approval/changes requested
4. **Suggestions** - Proposed code changes from reviewers

## Replying to Review Comments and Resolving Threads

### Replying to a Specific Review Comment

Line-level review comments have IDs. Retrieve them first, then post a reply:

```bash
# List all review comments with their IDs
gh api repos/OWNER/REPO/pulls/<PR_NUMBER>/comments | python3 -c "
import json, sys
data = json.load(sys.stdin)
for c in data:
    print(f'ID: {c[\"id\"]} | File: {c.get(\"path\")} | Body: {c[\"body\"][:100]}')
"

# Reply to a specific comment by ID
gh api repos/OWNER/REPO/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  --method POST \
  --field body="Done — added the trailing newline."
```

To get `OWNER` and `REPO` dynamically:
```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
# Then use: gh api repos/$REPO/pulls/<PR_NUMBER>/comments/...
```

### Resolving a Review Thread

Resolving threads requires the GraphQL API (the REST API does not expose this action):

```bash
# Step 1: Get thread IDs and their resolved state
gh api graphql -f query='
{
  repository(owner: "OWNER", name: "REPO") {
    pullRequest(number: PR_NUMBER) {
      reviewThreads(first: 20) {
        nodes {
          id
          isResolved
          comments(first: 1) {
            nodes {
              body
              path
            }
          }
        }
      }
    }
  }
}'

# Step 2: Resolve a specific thread by its GraphQL node ID
gh api graphql -f query='
mutation {
  resolveReviewThread(input: {threadId: "THREAD_NODE_ID"}) {
    thread {
      id
      isResolved
    }
  }
}'
```

### Typical Reply-and-Resolve Workflow

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
PR=9

# 1. See all open comments with IDs
gh api repos/$REPO/pulls/$PR/comments | python3 -c "
import json, sys
data = json.load(sys.stdin)
for c in data:
    print(f'ID: {c[\"id\"]} | {c.get(\"path\",\"\")} | {c[\"body\"][:80]}')
"

# 2. Fix the code/file as requested, commit and push
git add mononet/.editorconfig
git commit -m "Add trailing newline to .editorconfig as requested"
git push

# 3. Reply to each comment to confirm it was addressed
gh api repos/$REPO/pulls/$PR/comments/3072484363/replies \
  --method POST \
  --field body="Done — added the trailing newline."

# 4. Get GraphQL thread IDs to resolve threads
gh api graphql -f query='
{
  repository(owner: "davorrunje", name: "mononet") {
    pullRequest(number: 9) {
      reviewThreads(first: 20) {
        nodes { id isResolved comments(first:1) { nodes { body } } }
      }
    }
  }
}'

# 5. Resolve each thread
gh api graphql -f query='
mutation {
  resolveReviewThread(input: {threadId: "THREAD_NODE_ID"}) {
    thread { id isResolved }
  }
}'
```

## Handling Review Comments

### Common Review Requests

#### 1. Missing Newlines
**Problem:** Files should end with empty newline
```bash
# Fix: Add newline to end of file
echo "" >> filename
```

#### 2. Code Formatting Issues
**Problem:** Code doesn't follow project style
```bash
# Fix: Run project linting/formatting
./tools/lint.sh
# or
./tools/format.sh
```

#### 3. Missing Tests
**Problem:** New functionality needs test coverage
```bash
# Add tests in tests/ directory
# Run tests to verify
pytest
```

#### 4. Documentation Updates
**Problem:** Missing docstrings or documentation
- Add function/class docstrings
- Update README.md if needed
- Add examples for new features

#### 5. Type Hints
**Problem:** Missing type annotations
```python
# Before
def process_data(data):
    return data.upper()

# After
def process_data(data: str) -> str:
    return data.upper()
```

### Addressing Feedback Workflow

1. **Read all feedback** thoroughly
   ```bash
   gh pr view <PR_NUMBER> --comments
   ```

2. **Make requested changes** using appropriate tools
   - Edit files directly for simple changes
   - Use project tools for formatting/linting
   - Add tests as needed

3. **Test changes locally**
   ```bash
   # Run tests
   pytest

   # Run linting
   ./tools/lint.sh

   # Run static analysis
   ./tools/static-analysis.sh
   ```

4. **Commit and push updates**
   ```bash
   git add .
   git commit -m "Address review feedback: fix formatting and add tests"
   git push
   ```

5. **Respond to reviewers** (if needed)
   ```bash
   # Comment on PR to explain changes or ask questions
   gh pr comment <PR_NUMBER> --body "Fixed the formatting issues and added the missing newlines"
   ```

6. **Request re-review** (GitHub will auto-notify, but you can be explicit)
   ```bash
   gh pr review <PR_NUMBER> --comment --body "Ready for re-review"
   ```

## Fixing CI Failures

### Checking CI Status
```bash
# Check PR status including CI checks
gh pr status

# View detailed check results for a specific PR
gh pr checks <PR_NUMBER>

# View specific check details
gh pr view <PR_NUMBER> --json statusCheckRollup
```

### Common CI Failures and Solutions

#### 1. Test Failures
**Identifying:** CI shows "Tests failed" or specific test names
```bash
# Run tests locally to reproduce
pytest
pytest -v  # verbose output
pytest tests/specific_test.py::test_function  # specific test

# Check test coverage if required
pytest --cov=mononet
```

**Common fixes:**
- Fix failing assertions
- Update test data
- Add missing test dependencies
- Fix import issues

#### 2. Linting/Code Quality Failures
**Identifying:** CI shows "Lint check failed", "Code quality", or tool names (ruff, flake8, etc.)
```bash
# Run linting locally
./tools/lint.sh

# Or run specific tools
ruff check .
ruff format .
mypy mononet

# Fix automatically when possible
ruff check --fix .
ruff format .
```

**Common fixes:**
- Format code according to project standards
- Fix import order
- Remove unused variables/imports
- Add missing type hints
- Fix line length issues

#### 3. Build Failures
**Identifying:** CI shows "Build failed", "Installation failed"
```bash
# Test installation locally
pip install -e .

# Or use project build tools
uv sync
uv run pytest
```

**Common fixes:**
- Fix syntax errors
- Update dependencies in pyproject.toml
- Fix import paths
- Resolve circular imports

#### 4. Security/Static Analysis Failures
**Identifying:** CI shows "Security check failed", "bandit", "safety"
```bash
# Run security checks locally
./tools/static-analysis.sh

# Or run specific tools
bandit -r mononet
safety check
```

**Common fixes:**
- Remove hardcoded secrets/passwords
- Fix security vulnerabilities
- Update vulnerable dependencies
- Add security exclusions if false positives

#### 5. Coverage Failures
**Identifying:** CI shows "Coverage too low", "codecov"
```bash
# Check local coverage
pytest --cov=mononet --cov-report=html
open htmlcov/index.html  # View coverage report
```

**Common fixes:**
- Add tests for uncovered code
- Remove dead/unreachable code
- Add pragma comments for legitimate exclusions: `# pragma: no cover`

### CI Troubleshooting Workflow

1. **Check CI status and identify failing checks**
   ```bash
   gh pr checks <PR_NUMBER>
   ```

2. **Reproduce failures locally**
   ```bash
   # Run the same commands that CI runs
   ./tools/lint.sh
   ./tools/static-analysis.sh
   pytest
   ```

3. **Fix issues systematically**
   - Start with syntax errors and import issues
   - Fix linting and formatting
   - Address test failures
   - Handle security and coverage issues

4. **Test fixes locally before pushing**
   ```bash
   # Run full CI pipeline locally
   ./tools/lint.sh && ./tools/static-analysis.sh && pytest
   ```

5. **Commit and push fixes**
   ```bash
   git add .
   git commit -m "Fix CI failures: address linting and test issues"
   git push
   ```

6. **Wait for CI to re-run** (automatic) or **manually trigger** if needed

### Reading CI Logs and Details

#### Access Detailed Logs
```bash
# View check details with logs
gh pr view <PR_NUMBER> --json statusCheckRollup --jq '.statusCheckRollup[] | select(.conclusion=="FAILURE") | {name: .name, conclusion: .conclusion, detailsUrl: .detailsUrl}'

# For GitHub Actions specifically
gh run list --repo owner/repo --pr <PR_NUMBER>
gh run view <RUN_ID>
```

#### Understanding CI Error Messages
- **Exit code 1**: General failure, check specific tool output
- **Exit code 2**: Often linting/formatting issues
- **ModuleNotFoundError**: Import or dependency issues
- **AssertionError**: Test failures with specific assertions
- **SyntaxError**: Code syntax problems

### Preventing CI Failures

#### Pre-commit Setup
```bash
# Install and setup pre-commit hooks
pip install pre-commit
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

#### Local Testing Before Push
```bash
# Create a pre-push script
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
echo "Running pre-push checks..."
./tools/lint.sh && ./tools/static-analysis.sh && pytest
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Pre-push checks failed. Push cancelled."
    exit $exit_code
fi
echo "Pre-push checks passed!"
EOF

chmod +x .git/hooks/pre-push
```

### Example: Complete CI Failure Fix Workflow

```bash
# 1. Check what's failing
gh pr checks 123

# 2. See that linting and tests failed, reproduce locally
./tools/lint.sh  # Shows formatting issues
pytest           # Shows 2 failing tests

# 3. Fix linting issues
ruff format .
ruff check --fix .

# 4. Fix test failures
# Edit test files to fix assertions...

# 5. Verify fixes locally
./tools/lint.sh && pytest  # All pass now

# 6. Commit and push fixes
git add .
git commit -m "Fix CI failures: format code and fix failing tests

- Run ruff format to fix code formatting
- Fix assertion in test_user_validation
- Update test data for changed API response format"
git push

# 7. Verify CI passes on re-run
sleep 30  # Wait a bit for CI to start
gh pr checks 123  # Should show passing
```

## Advanced PR Management

### Handling Multiple Review Rounds
- Keep track of which comments have been addressed
- Use commit messages to reference specific feedback
- Group related fixes in single commits when possible

### Managing Large Change Requests
If reviewers request significant changes:

1. **Create a task list** to track all requests
2. **Make changes incrementally** with descriptive commits
3. **Test after each major change** to avoid breaking things
4. **Consider breaking into multiple PRs** if changes are extensive

### Resolving Conflicts
If your PR has merge conflicts:
```bash
# Update your branch with latest main
git checkout main
git pull
git checkout your-feature-branch
git merge main

# Resolve conflicts, then
git add .
git commit -m "Resolve merge conflicts"
git push
```

## Example: Complete Review Response Workflow

```bash
# 1. Check PR feedback
gh pr view 123 --comments

# 2. Save detailed comments for reference
gh api repos/owner/repo/pulls/123/comments > /tmp/review_feedback.json

# 3. Make the requested changes (example: add newlines)
echo "" >> mononet/.editorconfig
echo "" >> mononet/.gitattributes

# 4. Test changes
./tools/lint.sh

# 5. Commit fixes
git add .
git commit -m "Fix review feedback: add missing newlines to config files"

# 6. Push updates
git push

# 7. Comment on PR (optional)
gh pr comment 123 --body "Fixed the missing newlines in .editorconfig and .gitattributes"
```

## Example: Creating a New PR with Description File

```bash
# 1. Create comprehensive PR description
cat > /tmp/new_feature_pr.md << 'EOF'
# Add Missing Standard Files to Cookiecutter Template

This PR adds essential standard files that every professional repository should have.

## Files Added (10 total)
- **LICENSE** — PolyForm Noncommercial License 1.0.0
- **SECURITY.md** — Responsible disclosure policy
- **CONTRIBUTING.md** — Development and contribution guidelines
- **.editorconfig** — Consistent formatting configuration
- **.gitattributes** — Line endings and binary file handling
- **GitHub templates** — Issue and PR templates
- **CLAUDE.md** — AI development context

## Testing
- [x] Generated test project successfully
- [x] All files created with proper template variables
- [x] Template variables substituted correctly
- [x] Files follow best practices

## Related Issues
- Fixes SYN-8010

## Impact
Makes generated repositories complete and professional out-of-the-box.
EOF

# 2. Create the PR using the description file
gh pr create --title "Add missing standard files to cookiecutter template" --body-file /tmp/new_feature_pr.md

# 3. The PR is created with rich formatting and comprehensive details
```

## Making Subsequent Commits After Reviews

### Commit Message Best Practices for PRs
When making follow-up commits after review feedback, use descriptive commit messages:

```bash
# Good commit messages for review fixes
git commit -m "Address review feedback: add missing newlines to config files"
git commit -m "Fix linting issues: format code according to project standards"
git commit -m "Add tests for new validation function as requested"
git commit -m "Update documentation based on reviewer suggestions"

# Include specific details when helpful
git commit -m "Fix .editorconfig formatting

- Add trailing newline as requested by kumaranvpl
- Ensure file follows project standards"
```

### Commit Strategies During Review Process

#### Strategy 1: Incremental Commits (Recommended)
Make separate commits for each type of feedback:
```bash
# Fix formatting issues
git add .editorconfig .gitattributes
git commit -m "Fix config files: add missing trailing newlines"
git push

# Address code quality issues
git add src/
git commit -m "Fix linting issues: update imports and formatting"
git push

# Add requested tests
git add tests/
git commit -m "Add tests for user validation as requested"
git push
```

**Benefits:**
- Makes it easy for reviewers to see what changed
- Clear audit trail of fixes
- Can revert specific fixes if needed

#### Strategy 2: Squash Commits Later
Make incremental commits during development, squash before merge:
```bash
# During review process - make incremental commits
git commit -m "WIP: fix formatting"
git commit -m "WIP: add tests"
git commit -m "WIP: update docs"

# Before merge - squash related commits
git rebase -i HEAD~3  # Interactive rebase to squash
```

### Handling Complex Review Feedback

#### When Multiple Files Need Changes
Group related changes logically:
```bash
# Group 1: Configuration and formatting fixes
git add .editorconfig .gitattributes pyproject.toml
git commit -m "Config files: fix formatting and add missing settings

- Add trailing newlines to .editorconfig and .gitattributes
- Update pyproject.toml dependencies as requested
- Fix formatting to match project standards"

# Group 2: Code changes
git add src/ tests/
git commit -m "Code improvements: address review feedback

- Add type hints to validation functions
- Include tests for edge cases
- Fix imports and remove unused variables"

# Group 3: Documentation updates
git add README.md CONTRIBUTING.md docs/
git commit -m "Documentation: update based on review feedback

- Clarify installation instructions in README
- Add examples for new CLI options
- Update API documentation with new methods"
```

#### When Reviewers Request Major Changes
For significant architectural changes:

1. **Discuss first** - Comment on PR to clarify requirements
2. **Create a plan** - Outline the changes in PR comments
3. **Make changes incrementally** - One logical unit at a time
4. **Test between changes** - Avoid breaking the build
5. **Document the approach** - Explain your implementation decisions

```bash
# Example of major refactoring commits
git commit -m "Refactor: extract validation logic into separate module

This addresses the review feedback about code organization.
- Move validation functions to validators.py
- Update imports across the codebase
- Maintain backward compatibility"

git commit -m "Tests: update test structure to match new organization

- Move validation tests to test_validators.py
- Update test imports
- Add integration tests for refactored code"
```

## Updating PR Descriptions

### When to Update PR Descriptions
Update the PR description when:
- **Significant changes** are made during review
- **Scope expands** beyond original plan
- **Implementation approach changes**
- **New files** are added or removed
- **Testing approach** changes substantially

### How to Update PR Descriptions
```bash
# Method 1: Edit PR directly via GitHub CLI
gh pr edit <PR_NUMBER>  # Opens editor for title and body

# Method 2: Use a description file (recommended for complex updates)
cat > /tmp/updated_pr_description.md << 'EOF'
# Updated PR Title (if needed)

## Overview
Updated description reflecting changes made during review process.

## Changes Made
### Original Changes
- Original feature implementation
- Initial file additions

### Review Feedback Changes
- Fixed configuration file formatting (added trailing newlines)
- Updated documentation for clarity
- Added comprehensive PR workflow guide
- Renamed guide file for better scope description

## Files Modified
### Configuration Files
- `.editorconfig` — Added missing trailing newline
- `.gitattributes` — Added missing trailing newline

### Documentation
- `PULL_REQUEST_GUIDE.md` — Comprehensive PR workflow documentation (renamed from PR_REVIEW_GUIDE.md)
- `CLAUDE.md` — Updated references to new guide filename

## Testing
- [x] All original tests pass
- [x] Config files now pass linting checks
- [x] Documentation is comprehensive and accurate
- [x] File rename completed successfully

## Related Issues
- Fixes SYN-8010
- Addresses review feedback from kumaranvpl

## Review History
This PR has been updated based on reviewer feedback:
1. Added missing newlines to configuration files
2. Enhanced documentation scope and clarity
3. Renamed guide file for better descriptive accuracy
EOF

# Update the PR with the new description
gh pr edit <PR_NUMBER> --body-file /tmp/updated_pr_description.md
```

### PR Description Update Template
```markdown
# [Original/Updated Title]

## Overview
[Brief description, updated if scope changed]

## Changes Made
### Core Changes
- [Original planned changes]

### Review-Based Updates
- [Changes made due to feedback]
- [Additional improvements]

## Files [Added/Modified/Renamed]
### [Category 1]
- `file1.ext` — Description of changes made
- `file2.ext` — Description of changes made

### [Category 2]
- `file3.ext` — Description of changes made

## Testing
- [x] [Original testing completed]
- [x] [Review fixes tested]
- [x] [No regressions introduced]

## Review History
Brief summary of review rounds and major changes made.

## Related Issues
- Fixes SYN-XXXX
- Addresses feedback from [reviewer]
```

## Best Practices

### For AI-Generated PRs
- **Always review** AI-generated changes before pushing
- **Test thoroughly** as AI might miss edge cases
- **Provide context** to reviewers about AI involvement
- **Be responsive** to feedback and willing to iterate

### Communication
- **Be respectful** and professional in responses
- **Ask clarifying questions** if feedback is unclear
- **Explain your reasoning** for implementation choices
- **Thank reviewers** for their time and feedback

### Quality Assurance
- **Address all comments** before requesting re-review
- **Test changes comprehensively**
- **Follow project conventions** consistently
- **Update documentation** when adding features

## Troubleshooting

### Can't Access PR Comments
If GitHub CLI commands fail:
- Check you're authenticated: `gh auth status`
- Verify repo permissions: `gh repo view owner/repo`
- Try using full repo path: `gh pr view 123 --repo owner/repo`

### Comments Not Showing
- Ensure you're looking at the right PR number
- Check both issue comments and PR review comments
- Some comments may be on specific file lines

### Response Not Received
- Verify changes were pushed: `git status` + `git push`
- Check if PR is still open and reviewable
- Ensure reviewers are notified (GitHub usually handles this automatically)

---

This workflow ensures systematic handling of PR feedback and maintains high code quality while fostering good collaboration practices.
