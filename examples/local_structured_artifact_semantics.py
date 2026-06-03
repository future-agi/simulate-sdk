"""
Run a local structured-artifact semantic simulation.

This fixture behaves like a parsed receipt, form, table, or log export. The
agent inspects the structured artifact, and ai-evaluation checks exact fields,
line-item rows, event sequence, and answer claims without a model judge.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    Persona,
    Scenario,
    StructuredArtifactEnvironment,
    TestRunner,
    evaluate_agent_report,
)


def receipt_fixture():
    return {
        "receipt_123": {
            "domain": "receipt",
            "schema": "receipt_v1",
            "description": "Parsed receipt for order 123.",
            "data": {
                "receipt_id": "rcpt_123",
                "merchant": "Northwind",
                "order": {"id": "123"},
                "total": {"amount": 42.0, "currency": "USD"},
                "line_items": [
                    {"sku": "SKU-1", "description": "Widget", "quantity": 2, "amount": 20.0},
                    {"sku": "TAX", "description": "Tax", "quantity": 1, "amount": 2.0},
                ],
                "events": [
                    {"event": "created"},
                    {"event": "paid"},
                    {"event": "captured"},
                ],
            },
        }
    }


async def structured_receipt_agent(input):
    return AgentResponse(
        content="Receipt rcpt_123 from Northwind has total $42.00 and SKU-1 quantity 2.",
        tool_calls=[
            {"id": "list", "name": "list_structured_artifacts", "arguments": {}},
            {
                "id": "inspect",
                "name": "inspect_structured_artifact",
                "arguments": {"id": "receipt_123"},
            },
        ],
    )


async def main():
    scenario = Scenario(
        name="structured-artifact-semantics",
        dataset=[
            Persona(
                persona={"name": "Anika", "risk_profile": "standard"},
                situation="Anika needs a parsed receipt checked before a refund.",
                outcome="Receipt rcpt_123 from Northwind totals $42.00 with SKU-1 quantity 2.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=structured_receipt_agent,
        environment=StructuredArtifactEnvironment(receipt_fixture(), default_domain="receipt"),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config={
            "required_tools": ["list_structured_artifacts", "inspect_structured_artifact"],
            "available_tools": ["list_structured_artifacts", "inspect_structured_artifact"],
            "required_artifact_types": ["json"],
            "artifact_semantic_checks": [
                {
                    "id": "receipt_semantics",
                    "artifact": {
                        "type": "json",
                        "id": "receipt_123",
                        "metadata": {"domain": "receipt", "schema": "receipt_v1"},
                    },
                    "expected_fields": {
                        "receipt_id": "rcpt_123",
                        "merchant": "Northwind",
                        "order.id": "123",
                        "total.amount": 42.0,
                        "total.currency": "USD",
                    },
                    "answer_fields": {
                        "receipt_id": ["rcpt_123"],
                        "merchant": ["Northwind"],
                        "total.amount": ["$42.00"],
                    },
                    "required_rows": [
                        {
                            "path": "line_items",
                            "where": {"sku": "SKU-1"},
                            "fields": {"quantity": 2, "amount": 20.0},
                        }
                    ],
                    "event_sequence": {
                        "path": "events",
                        "field": "event",
                        "expected": ["created", "paid", "captured"],
                    },
                    "forbidden_answer_terms": ["$24.00", "SKU-9"],
                }
            ],
            "metric_weights": {"artifact_semantics_quality": 5.0, "artifact_coverage": 1.0},
        },
        threshold=0.85,
    )

    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("artifact_coverage:", metrics.get("artifact_coverage"))
    print("artifact_semantics_quality:", metrics.get("artifact_semantics_quality"))


if __name__ == "__main__":
    asyncio.run(main())
