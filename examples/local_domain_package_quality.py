"""
Run a local domain-package workflow simulation.

This fixture behaves like real workflow packages: a support ticket, a refund
ledger, a QA calendar, and an email thread. `ai-evaluation` checks package-level
invariants such as status/assignee/SLA, balanced ledger entries, calendar
conflicts, chronological thread order, and participant coverage without a model
judge.

Requires:
    pip install agent-simulate ai-evaluation
"""

import asyncio

from fi.simulate import (
    AgentResponse,
    DomainPackageEnvironment,
    Persona,
    Scenario,
    TestRunner,
    evaluate_agent_report,
)


def package_fixture():
    return {
        "ticket_123": {
            "domain": "support",
            "package_type": "support_ticket",
            "description": "Refund support ticket package.",
            "data": {
                "ticket_id": "TCK-123",
                "status": "resolved",
                "assignee": {"id": "agent_priya", "name": "Priya"},
                "sla": {"met": True},
            },
        },
        "ledger_9": {
            "domain": "finance",
            "package_type": "ledger",
            "description": "Refund ledger package.",
            "data": {
                "ledger_id": "LED-9",
                "entries": [
                    {"account": "refunds", "debit": 42.0, "credit": 0.0},
                    {"account": "cash", "debit": 0.0, "credit": 42.0},
                ],
            },
        },
        "qa_calendar": {
            "domain": "calendar",
            "package_type": "calendar",
            "description": "QA review schedule.",
            "data": {
                "events": [
                    {
                        "id": "handoff",
                        "start": "2026-06-03T10:00:00",
                        "end": "2026-06-03T10:30:00",
                        "participants": ["agent_priya"],
                    },
                    {
                        "id": "qa",
                        "start": "2026-06-03T10:30:00",
                        "end": "2026-06-03T11:00:00",
                        "participants": ["agent_priya"],
                    },
                ]
            },
        },
        "thread_refund": {
            "domain": "email",
            "package_type": "email_thread",
            "description": "Customer refund thread.",
            "data": {
                "messages": [
                    {
                        "sent_at": "2026-06-03T09:00:00",
                        "from": "avery@example.com",
                        "to": ["priya@example.com"],
                    },
                    {
                        "sent_at": "2026-06-03T09:05:00",
                        "from": "priya@example.com",
                        "to": ["avery@example.com"],
                    },
                ]
            },
        },
    }


async def domain_package_agent(input):
    return AgentResponse(
        content=(
            "Ticket TCK-123 is resolved by Priya. Ledger LED-9 is balanced, "
            "QA calendar has no conflict, and the thread includes Avery and Priya."
        ),
        tool_calls=[
            {"id": "list", "name": "list_domain_packages", "arguments": {}},
            {"id": "ticket", "name": "inspect_domain_package", "arguments": {"id": "ticket_123"}},
            {"id": "ledger", "name": "inspect_domain_package", "arguments": {"id": "ledger_9"}},
            {"id": "calendar", "name": "inspect_domain_package", "arguments": {"id": "qa_calendar"}},
            {"id": "thread", "name": "inspect_domain_package", "arguments": {"id": "thread_refund"}},
        ],
    )


def domain_package_config():
    return {
        "required_tools": ["list_domain_packages", "inspect_domain_package"],
        "available_tools": ["list_domain_packages", "inspect_domain_package"],
        "required_artifact_types": ["json"],
        "domain_package_checks": [
            {
                "id": "support_ticket_package",
                "package_id": "ticket_123",
                "domain": "support",
                "package_type": "support_ticket",
                "expected_fields": {"ticket_id": "TCK-123", "status": "resolved", "sla.met": True},
                "answer_fields": {"ticket_id": ["TCK-123"], "assignee.name": ["Priya"]},
                "invariants": [
                    {"type": "field_present", "path": "assignee.id"},
                    {"type": "status_in", "path": "status", "allowed": ["resolved", "closed"]},
                    {"type": "field_equals", "path": "sla.met", "value": True},
                ],
            },
            {
                "id": "ledger_package",
                "package_id": "ledger_9",
                "domain": "finance",
                "package_type": "ledger",
                "invariants": [{"type": "ledger_balanced", "entries_path": "entries"}],
            },
            {
                "id": "calendar_package",
                "package_id": "qa_calendar",
                "domain": "calendar",
                "package_type": "calendar",
                "invariants": [{"type": "calendar_no_overlap", "events_path": "events"}],
            },
            {
                "id": "thread_package",
                "package_id": "thread_refund",
                "domain": "email",
                "package_type": "email_thread",
                "invariants": [
                    {"type": "chronological", "items_path": "messages", "time_field": "sent_at"},
                    {
                        "type": "required_participants",
                        "items_path": "messages",
                        "participants": ["avery@example.com", "priya@example.com"],
                        "item_participant_paths": ["from", "to"],
                    },
                ],
            },
        ],
        "metric_weights": {"domain_package_quality": 6.0, "artifact_coverage": 1.0},
    }


async def main():
    scenario = Scenario(
        name="domain-package-quality",
        dataset=[
            Persona(
                persona={"name": "Avery", "risk_profile": "standard"},
                situation="Avery needs a refund support package closed correctly.",
                outcome="Support ticket, ledger, calendar, and email thread package checks pass.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=domain_package_agent,
        environment=DomainPackageEnvironment(package_fixture(), default_domain="support"),
        max_turns=1,
        min_turns=1,
    )
    evaluation = evaluate_agent_report(
        report,
        config=domain_package_config(),
        threshold=0.85,
    )

    metrics = evaluation.summary["metric_averages"]
    print("score:", evaluation.score)
    print("passed:", evaluation.passed)
    print("artifact_coverage:", metrics.get("artifact_coverage"))
    print("domain_package_quality:", metrics.get("domain_package_quality"))


if __name__ == "__main__":
    asyncio.run(main())
