"""
Run local simulation against a versioned domain-package registry.

`DomainPackageEnvironment` emits an enterprise claim package, and
`ai-evaluation` scores it through a local Future AGI registry config with a
custom alias, required fields, required documents, status values, and numeric
tolerance.

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


DOMAIN_PACKAGE_REGISTRY = {
    "version": "futureagi.domain-packages.acme.v1",
    "presets": {
        "claim_file": {
            "version": "acme-claims-2026-06",
            "aliases": ["enterprise_claim"],
            "required_fields": ["adjuster.id"],
            "invariants": [
                {
                    "type": "collection_contains",
                    "items_path": "documents",
                    "field": "type",
                    "values_key": "claim_audit_documents",
                    "default_values": ["audit_trail"],
                }
            ],
        }
    },
}


def package_fixture():
    return {
        "enterprise_claim_9": {
            "domain": "insurance",
            "package_type": "enterprise_claim",
            "description": "Enterprise claim packet with custom registry requirements.",
            "data": {
                "claim_id": "ECLM-9",
                "status": "review_complete",
                "claimant": {"id": "cust_9"},
                "adjuster": {"id": "adj_1"},
                "loss": {"date": "2026-06-01"},
                "coverage": {"limit": 1000.0},
                "amount": 1020.0,
                "documents": [
                    {"type": "loss_notice"},
                    {"type": "policy"},
                    {"type": "audit_trail"},
                ],
            },
        }
    }


async def domain_package_agent(input):
    return AgentResponse(
        content="Enterprise claim ECLM-9 is review complete.",
        tool_calls=[
            {"id": "list", "name": "list_domain_packages", "arguments": {}},
            {
                "id": "claim",
                "name": "inspect_domain_package",
                "arguments": {"id": "enterprise_claim_9"},
            },
        ],
    )


def domain_package_config():
    return {
        "required_tools": ["list_domain_packages", "inspect_domain_package"],
        "available_tools": ["list_domain_packages", "inspect_domain_package"],
        "required_artifact_types": ["json"],
        "domain_package_registry": DOMAIN_PACKAGE_REGISTRY,
        "domain_package_checks": [
            {
                "id": "enterprise_claim_preset",
                "package_id": "enterprise_claim_9",
                "package_type": "enterprise_claim",
                "allowed_statuses": ["review_complete"],
                "amount_tolerance": 25.0,
                "claim_audit_documents": ["audit_trail"],
            }
        ],
        "metric_weights": {"domain_package_quality": 6.0, "artifact_coverage": 1.0},
    }


async def main():
    scenario = Scenario(
        name="domain-package-registry",
        dataset=[
            Persona(
                persona={"name": "Avery", "risk_profile": "standard"},
                situation="Avery needs an enterprise claim packet validated.",
                outcome="The enterprise claim satisfies the configured registry preset.",
            )
        ],
    )
    report = await TestRunner().run_test(
        scenario=scenario,
        agent_callback=domain_package_agent,
        environment=DomainPackageEnvironment(package_fixture(), default_domain="insurance"),
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
