"""
Run a local evidence-grounding simulation.

This combines a retrieval/memory trace with an image artifact fixture, then
scores whether the final answer avoids source contradictions and grounds a
receipt-total claim in OCR-like artifact metadata.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    ImageEnvironment,
    Persona,
    RetrievalMemoryEnvironment,
    Scenario,
    TestRunner,
    evaluate_agent_report,
)


async def evidence_grounded_agent(input):
    return AgentResponse(
        content="Order 123 has a 30 day refund window. The receipt total is $42.00.",
        tool_calls=[
            {
                "id": "search",
                "name": "search_knowledge_base",
                "arguments": {"query": "refund policy order 123", "top_k": 1},
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
                    "claim": "Order 123 has a 30 day refund window.",
                    "freshness_checked": True,
                },
            },
            {"id": "images", "name": "list_images", "arguments": {}},
            {"id": "receipt", "name": "inspect_image", "arguments": {"id": "receipt_123"}},
            {"id": "status", "name": "retrieval_memory_status", "arguments": {}},
        ],
    )


async def main():
    scenario = Scenario(
        name="evidence-grounding",
        dataset=[
            Persona(
                persona={"name": "Anika", "risk_profile": "standard"},
                situation="Anika needs refund policy and receipt evidence checked.",
                outcome="Order 123 has a 30 day refund window and a $42.00 receipt total.",
            )
        ],
    )
    retrieval = RetrievalMemoryEnvironment(
        {
            "refund_policy_current": {
                "title": "Refund Policy v2",
                "content": "Order 123 has a 30 day refund window and no restocking fee.",
                "source": "policy.md",
                "version": "v2",
                "current": True,
            }
        }
    )
    images = ImageEnvironment(
        {
            "receipt_123": {
                "description": "Receipt for order 123.",
                "metadata": {"ocr_text": "Receipt order 123 total $42.00 paid by card."},
            }
        }
    )

    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=evidence_grounded_agent,
        environment=[retrieval, images],
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": [
                "search_knowledge_base",
                "read_document",
                "cite_sources",
                "list_images",
                "inspect_image",
                "retrieval_memory_status",
            ],
            "available_tools": [
                "search_knowledge_base",
                "read_document",
                "cite_sources",
                "list_images",
                "inspect_image",
                "retrieval_memory_status",
            ],
            "required_artifact_types": ["trace", "image"],
            "source_contradiction_checks": [
                {
                    "id": "refund_window",
                    "source_terms": ["30 day refund window"],
                    "answer_terms": ["refund window"],
                    "contradict_terms": ["90 day refund window", "non refundable"],
                }
            ],
            "artifact_grounding_checks": [
                {
                    "id": "receipt_total",
                    "artifact": {"type": "image", "id": "receipt_123"},
                    "answer_terms": ["receipt total", "$42.00"],
                    "support_terms": ["total $42.00"],
                    "forbidden_answer_terms": ["$24.00"],
                }
            ],
            "metric_weights": {
                "source_contradiction": 4.0,
                "artifact_grounding_quality": 4.0,
            },
        },
        threshold=0.85,
    )

    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("source_contradiction:", metrics.get("source_contradiction"))
    print("artifact_grounding_quality:", metrics.get("artifact_grounding_quality"))


if __name__ == "__main__":
    asyncio.run(main())
