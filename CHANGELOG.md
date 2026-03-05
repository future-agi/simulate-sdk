# Changelog

## 0.1.3

### Fixed
- Cloud Engine: Update call execution status to 'completed' or 'failed' upon agent call failure
  - Sets status to 'completed' if failure occurs after at least one turn (allows evaluations to run on partial data)
  - Sets status to 'failed' if failure occurs on the first turn
  - Suppresses success log message when agent call has failed
  - Uses generic error message to avoid leaking internal error details

## 0.1.2

### Added
- Cloud Mode: Added backend API integration for orchestrating simulations via HTTP
- Agent Wrappers: Created framework-specific wrappers for OpenAI, LangChain, Gemini, and Anthropic
- Tool Calls Support: Added `tool_calls` and `tool_responses` fields to `AgentResponse` for structured tool data
- Run Test Name Support: Added `run_test_name` parameter as an alternative to `run_id` for better developer experience
- Timeout Configuration: Added configurable timeout parameter for API requests (default: 120s)
- OpenTelemetry Integration: Added simulator attribute propagation via baggage context using `fi_instrumentation`
- Test Suite: Added comprehensive tests for agent wrappers and runner dispatch logic

### Changed
- Decoupled LiveKit: Moved all LiveKit-specific logic to `LiveKitEngine`, making it an optional dependency
- Architecture: Refactored `TestRunner` to use Strategy Pattern with `BaseEngine`, `LiveKitEngine`, and `CloudEngine`
- Imports: Converted all relative imports to absolute imports (`from fi.simulate...`)
- Error Handling: Improved error messages to display backend's actual error response instead of generic HTTP status codes
- Role Conversion: Implemented role mapping between SDK and backend (SDK "assistant" → backend "user", backend "assistant" → SDK "user")
- Message Filtering: Filter out `tool` and `system` messages and empty content from conversation history sent to user's agent
