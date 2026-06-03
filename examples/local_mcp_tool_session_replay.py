"""
Replay an MCP tool-session export locally.

This models tools/list and tools/call evidence from an MCP server export,
including tool schemas, arguments, structured results, and errors. It does not
start an MCP server, call a model, or require an API key.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
    load_mcp_tool_session_export,
)


MCP_SESSION_EXPORT = [
    {
        "jsonrpc": "2.0",
        "id": "list_1",
        "result": {
            "tools": [
                {
                    "name": "search_order",
                    "description": "Look up an order by id.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                        "additionalProperties": False,
                    },
                }
            ]
        },
    },
    {
        "jsonrpc": "2.0",
        "id": "call_1",
        "method": "tools/call",
        "params": {"name": "search_order", "arguments": {"order_id": "ord_123"}},
    },
    {
        "jsonrpc": "2.0",
        "id": "call_1",
        "result": {"structuredContent": {"resolved": True, "status": "found"}},
    },
]


async def mcp_trace_agent(input):
    return AgentResponse(
        content="MCP tool session inspected with schema and result evidence.",
        tool_calls=[
            {"id": "status", "name": "framework_trace_status", "arguments": {}},
            {
                "id": "schemas",
                "name": "list_framework_spans",
                "arguments": {"signal": "mcp_tool_schema"},
            },
            {
                "id": "results",
                "name": "list_framework_spans",
                "arguments": {"signal": "mcp_tool_result"},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="mcp-tool-session-replay",
        dataset=[
            Persona(
                persona={"name": "Avery", "risk_profile": "standard"},
                situation="Avery needs an MCP support-tool session inspected before optimization.",
                outcome="The MCP session is inspected with tool schema and result evidence.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=mcp_trace_agent,
        environment=load_mcp_tool_session_export(
            MCP_SESSION_EXPORT,
            server_name="support-tools",
        ),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "framework_trace_status",
                "list_framework_spans",
            ],
            "available_tools": [
                "framework_trace_status",
                "list_framework_spans",
            ],
            "required_artifact_types": ["trace"],
            "required_framework_trace": [
                "tool",
                "mcp_tool_schema",
                "mcp_tool_call",
                "mcp_tool_result",
            ],
            "expected_tool_outcomes": {
                "search_order": {
                    "success": True,
                    "result": {"resolved": True, "status": "found"},
                }
            },
            "success_criteria": ["MCP tool session inspected"],
        },
        threshold=0.9,
    )

    trace_state = report.results[0].metadata["environment_state"]["framework_trace"]
    metrics = evaluation.summary["metric_averages"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("tools:", trace_state["metadata"]["mcp_tool_session"]["tool_names"])
    print("signals:", trace_state["signals"])
    print("framework_trace_coverage:", metrics["framework_trace_coverage"])
    print("tool_argument_schema:", metrics["tool_argument_schema"])
    print("tool_outcome:", metrics["tool_outcome"])


if __name__ == "__main__":
    asyncio.run(main())
