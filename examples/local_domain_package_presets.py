"""
Run local domain-package preset simulation.

`DomainPackageEnvironment` exposes claim, contract, CRM, procurement, clinical
intake, and incident-response fixtures. `ai-evaluation` scores their preset
invariants locally through the Future AGI eval stack, without external keys.

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
        "claim_9": {
            "domain": "insurance",
            "package_type": "insurance_claim",
            "description": "Claim file awaiting package validation.",
            "data": {
                "claim_id": "CLM-9",
                "status": "approved",
                "claimant": {"id": "cust_9"},
                "loss": {"date": "2026-06-01"},
                "coverage": {"limit": 1000.0},
                "amount": 875.0,
                "documents": [{"type": "loss_notice"}, {"type": "policy"}],
            },
        },
        "contract_7": {
            "domain": "legal",
            "package_type": "contract_review",
            "description": "Contract review packet.",
            "data": {
                "contract_id": "CTR-7",
                "effective_date": "2026-06-01",
                "expiration_date": "2027-06-01",
                "parties": [{"id": "acme"}, {"id": "futureagi"}],
                "signatures": [
                    {"party_id": "acme", "status": "signed"},
                    {"party_id": "futureagi", "status": "executed"},
                ],
            },
        },
        "account_acme": {
            "domain": "crm",
            "package_type": "crm_account_plan",
            "description": "CRM account plan for ACME.",
            "data": {
                "account_id": "ACME",
                "owner": {"id": "owner_1"},
                "last_touch_at": "2026-06-01T09:00:00",
                "next_step": {
                    "action": "security review",
                    "due_at": "2026-06-05T09:00:00",
                },
                "contacts": [{"id": "c1", "role": "economic_buyer"}],
            },
        },
        "po_8": {
            "domain": "procurement",
            "package_type": "purchase_order",
            "description": "Procurement purchase order.",
            "data": {
                "po_id": "PO-8",
                "status": "approved",
                "vendor": {"id": "vendor_1"},
                "line_items": [
                    {"sku": "A", "quantity": 2, "unit_price": 50.0},
                    {"sku": "B", "quantity": 1, "unit_price": 140.0},
                ],
                "total": 240.0,
                "approvals": [
                    {"role": "requester", "status": "approved"},
                    {"role": "finance", "status": "approved"},
                ],
            },
        },
        "clinical_4": {
            "domain": "clinical",
            "package_type": "clinical_intake",
            "description": "Clinical intake packet.",
            "data": {
                "patient": {"id": "pat_4"},
                "encounter": {"reason": "knee pain"},
                "consent": {"signed_at": "2026-06-03T08:00:00"},
                "triage": {"level": "urgent"},
                "sections": [
                    {"name": "allergies"},
                    {"name": "medications"},
                    {"name": "consent"},
                ],
            },
        },
        "incident_5": {
            "domain": "security",
            "package_type": "incident_response",
            "description": "Incident response package.",
            "data": {
                "incident_id": "INC-5",
                "severity": "high",
                "status": "contained",
                "detected_at": "2026-06-03T10:00:00",
                "contained_at": "2026-06-03T10:45:00",
                "owner": {"id": "sec_1"},
                "actions": [{"type": "containment"}, {"type": "customer_update"}],
            },
        },
    }


async def domain_package_agent(input):
    return AgentResponse(
        content=(
            "Claim CLM-9, contract CTR-7, account ACME, purchase order PO-8, "
            "clinical intake INT-4, and incident INC-5 are ready for review."
        ),
        tool_calls=[
            {"id": "list", "name": "list_domain_packages", "arguments": {}},
            {"id": "claim", "name": "inspect_domain_package", "arguments": {"id": "claim_9"}},
            {"id": "contract", "name": "inspect_domain_package", "arguments": {"id": "contract_7"}},
            {"id": "account", "name": "inspect_domain_package", "arguments": {"id": "account_acme"}},
            {"id": "po", "name": "inspect_domain_package", "arguments": {"id": "po_8"}},
            {"id": "clinical", "name": "inspect_domain_package", "arguments": {"id": "clinical_4"}},
            {"id": "incident", "name": "inspect_domain_package", "arguments": {"id": "incident_5"}},
        ],
    )


def domain_package_config():
    return {
        "required_tools": ["list_domain_packages", "inspect_domain_package"],
        "available_tools": ["list_domain_packages", "inspect_domain_package"],
        "required_artifact_types": ["json"],
        "domain_package_checks": [
            {"id": "claim_preset", "package_id": "claim_9", "package_type": "insurance_claim"},
            {"id": "contract_preset", "package_id": "contract_7", "package_type": "contract_review"},
            {"id": "crm_preset", "package_id": "account_acme", "package_type": "crm_account_plan"},
            {"id": "po_preset", "package_id": "po_8", "package_type": "purchase_order"},
            {"id": "clinical_preset", "package_id": "clinical_4", "package_type": "clinical_intake"},
            {"id": "incident_preset", "package_id": "incident_5", "package_type": "incident_response"},
        ],
        "metric_weights": {"domain_package_quality": 6.0, "artifact_coverage": 1.0},
    }


async def main():
    scenario = Scenario(
        name="domain-package-presets",
        dataset=[
            Persona(
                persona={"name": "Avery", "risk_profile": "standard"},
                situation="Avery needs six business workflow packages validated.",
                outcome="Every package-family preset passes deterministic invariant checks.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=domain_package_agent,
        environment=DomainPackageEnvironment(package_fixture(), default_domain="operations"),
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
