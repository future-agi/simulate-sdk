class ManifestLangGraphAgent:
    async def ainvoke(self, payload):
        return {
            "content": "Manifest-declared LangGraph runtime passed with trace evidence.",
            "tool_calls": [
                {
                    "id": "policy",
                    "name": "lookup_policy",
                    "arguments": {"topic": "refund"},
                }
            ],
            "metadata": {"runtime_contract": {"passed": True}},
            "events": [
                {
                    "type": "runtime_checkpoint",
                    "name": "adapter_contract",
                    "payload": {
                        "method": "ainvoke",
                        "input_mode": "dict",
                        "payload_keys": sorted(payload.keys()),
                    },
                }
            ],
            "artifacts": [
                {
                    "type": "json",
                    "role": "assistant",
                    "data": {
                        "contract": "ok",
                        "framework": payload["metadata"]["framework"],
                    },
                    "metadata": {"kind": "runtime_contract"},
                }
            ],
        }


def build_agent():
    return ManifestLangGraphAgent()
