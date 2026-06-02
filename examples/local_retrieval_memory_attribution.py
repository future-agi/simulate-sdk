"""
Run a local retrieval and memory attribution simulation.

This models RAG and agent-memory evidence: retrieval queries, returned source
documents, document reads, memory reads/writes, freshness/version metadata, and
citations that attribute claims to sources. No vector database, model, or cloud
service is required.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    Persona,
    RetrievalMemoryEnvironment,
    Scenario,
    TestRunner,
    evaluate_agent_report,
)


async def grounded_memory_agent(input):
    return AgentResponse(
        content=(
            "Order 123 is eligible for refund based on the current policy and "
            "remembered order context."
        ),
        tool_calls=[
            {
                "id": "search",
                "name": "search_knowledge_base",
                "arguments": {"query": "current refund policy order 123", "top_k": 2},
            },
            {
                "id": "memory_read",
                "name": "retrieve_memory",
                "arguments": {"key": "order_id"},
            },
            {
                "id": "read_policy",
                "name": "read_document",
                "arguments": {"id": "refund_policy_current"},
            },
            {
                "id": "cite",
                "name": "cite_sources",
                "arguments": {
                    "doc_ids": ["refund_policy_current"],
                    "memory_keys": ["order_id"],
                    "claim": "Order 123 is eligible for refund.",
                    "freshness_checked": True,
                },
            },
            {
                "id": "memory_write",
                "name": "write_memory",
                "arguments": {"key": "last_resolution", "value": "refund eligible"},
            },
            {"id": "status", "name": "retrieval_memory_status", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="retrieval-memory-attribution",
        dataset=[
            Persona(
                persona={"name": "Anika", "risk_profile": "standard"},
                situation="Anika needs a refund answer grounded in current policy.",
                outcome="Order 123 is eligible for refund based on current policy.",
            )
        ],
    )
    environment = RetrievalMemoryEnvironment(
        {
            "refund_policy_current": {
                "title": "Refund Policy v2",
                "content": "Order 123 can be refunded when policy approval is current.",
                "source": "policy.md",
                "version": "v2",
                "current": True,
                "metadata": {"last_modified": "2026-05-01"},
            },
            "refund_policy_old": {
                "title": "Refund Policy v1",
                "content": "Old refund rules for order 123.",
                "source": "policy-old.md",
                "version": "v1",
                "current": False,
                "metadata": {"last_modified": "2024-01-01"},
            },
        },
        memory={"order_id": "123", "customer_tier": "gold"},
    )

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=grounded_memory_agent,
        environment=environment,
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "search_knowledge_base",
                "retrieve_memory",
                "read_document",
                "cite_sources",
                "write_memory",
                "retrieval_memory_status",
            ],
            "available_tools": [
                "search_knowledge_base",
                "retrieve_memory",
                "read_document",
                "cite_sources",
                "write_memory",
                "retrieval_memory_status",
            ],
            "required_artifact_types": ["trace"],
            "required_retrieval_memory_trace": [
                "query",
                "document",
                "memory_read",
                "memory_write",
                "citation",
                "attribution",
                "freshness",
            ],
            "memory_allowed_keys": ["order_id", "customer_tier", "last_resolution"],
            "success_criteria": ["eligible for refund"],
        },
        threshold=0.85,
    )

    result = report.results[0]
    trace_state = result.metadata["environment_state"]["retrieval_memory"]

    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("queries:", trace_state["queries"])
    print("citations:", trace_state["citations"])
    print(
        "retrieval_memory_attribution:",
        evaluation.summary["metric_averages"]["retrieval_memory_attribution"],
    )


if __name__ == "__main__":
    asyncio.run(main())
