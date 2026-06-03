"""
Run a local browser/CUA replay with semantic and masked visual-diff regions.

Requires:
    pip install agent-simulate ai-evaluation

The example writes two tiny PNG screenshots, lets BrowserEnvironment compare
them after an action, and scores whether only the allowed semantic status
region changed after masking dynamic clock noise.
"""

import asyncio
import struct
import tempfile
import zlib
from pathlib import Path

from fi.simulate import AgentResponse, BrowserEnvironment, Persona, Scenario, TestRunner, evaluate_agent_report


def write_png(path: Path, width: int, height: int, pixels) -> None:
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row.extend(pixels[y][x])
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def build_screenshots(directory: Path) -> tuple[Path, Path]:
    before_path = directory / "checkout-before.png"
    after_path = directory / "checkout-after.png"
    white = (255, 255, 255, 255)
    green = (20, 180, 80, 255)
    blue = (30, 120, 240, 255)
    before_pixels = [[white for _ in range(4)] for _ in range(4)]
    after_pixels = [[white for _ in range(4)] for _ in range(4)]
    after_pixels[0][0] = blue
    for y in (1, 2):
        for x in (1, 2):
            after_pixels[y][x] = green
    write_png(before_path, 4, 4, before_pixels)
    write_png(after_path, 4, 4, after_pixels)
    return before_path, after_path


async def browser_agent(input):
    return AgentResponse(
        content="I clicked confirm and captured the semantic visual change.",
        tool_calls=[
            {
                "id": "click",
                "name": "browser_click",
                "arguments": {"selector": "#confirm", "action": "click confirm"},
            }
        ],
    )


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        before_path, after_path = build_screenshots(Path(tmpdir))
        scenario = Scenario(
            name="browser-semantic-visual-diff",
            dataset=[
                Persona(
                    persona={"name": "Riya", "risk_profile": "standard"},
                    situation="Riya needs checkout confirmed without counting dynamic clock pixels.",
                    outcome="Only the semantic status banner changes after masking the clock.",
                )
            ],
        )
        report = await TestRunner().run_test(
            scenario=scenario,
            agent_callback=browser_agent,
            environment=BrowserEnvironment(
                url="https://shop.example.com/checkout",
                dom="<button id='confirm'>Confirm</button><output id='status'>Pending</output>",
                screenshot_uri=f"file://{before_path}",
                allowed_domains=["shop.example.com"],
                regions={
                    "session_clock": {
                        "x": 0,
                        "y": 0,
                        "width": 1,
                        "height": 1,
                        "role": "timer",
                        "text": "10:01",
                        "masked": True,
                    },
                    "status_banner": {
                        "x": 1,
                        "y": 1,
                        "width": 2,
                        "height": 2,
                        "selector": "#status",
                        "role": "status",
                        "text": "Confirmed",
                        "allowed_change": True,
                    },
                    "total_due": {
                        "x": 3,
                        "y": 3,
                        "width": 1,
                        "height": 1,
                        "role": "amount",
                        "text": "$42.00",
                        "forbidden_change": True,
                    },
                },
                actions=[
                    {
                        "id": "confirm_semantic_change",
                        "tool_names": ["browser_click"],
                        "selector": "#confirm",
                        "screenshot_path": str(after_path),
                        "screenshot_diff": {
                            "id": "confirm_semantic_delta",
                            "threshold": 0,
                            "semantic_regions": ["status_banner"],
                            "allowed_regions": ["status_banner"],
                            "masked_regions": ["session_clock"],
                            "forbidden_regions": ["total_due"],
                        },
                    }
                ],
            ),
            max_turns=1,
            min_turns=1,
            modality="cua",
        )
        evaluation = evaluate_agent_report(
            report,
            config={
                "required_tools": ["browser_click"],
                "available_tools": ["browser_click"],
                "required_artifact_types": ["trace", "screenshot"],
                "required_browser_trace": ["semantic_screenshot_diff", "masked_screenshot_diff"],
                "expected_browser_screenshot_diffs": [
                    {
                        "id": "confirm_semantic_delta",
                        "semantic_regions": ["status_banner"],
                        "allowed_regions": ["status_banner"],
                        "masked_regions": ["session_clock"],
                        "forbidden_regions": ["total_due"],
                        "only_allowed_regions_changed": True,
                    }
                ],
            },
            threshold=0.85,
        )
        result = report.results[0]
        browser = result.metadata["environment_state"]["browser"]
        summary = browser["screenshot_diffs"][-1]["semantic_summary"]
        metrics = evaluation.summary["metric_averages"]
        print("score:", evaluation.score)
        print("passed:", evaluation.passed)
        print("changed_regions:", summary["changed_regions"])
        print("masked_changed_regions:", summary["masked_changed_regions"])
        print("effective_changed_regions:", summary["effective_changed_regions"])
        print("only_allowed_regions_changed:", summary["only_allowed_regions_changed"])
        print("browser_trace_coverage:", metrics.get("browser_trace_coverage"))
        print("browser_grounding_quality:", metrics.get("browser_grounding_quality"))


if __name__ == "__main__":
    asyncio.run(main())
