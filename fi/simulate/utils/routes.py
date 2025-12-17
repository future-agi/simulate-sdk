import httpx
import os
from typing import Dict, Any, List, Optional

class APIRoutes:
    """
    Handles API interactions with the Future AGI backend.
    """
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "x-api-key": self.api_key,
            "x-secret-key": self.secret_key,
            "Content-Type": "application/json"
        }
        # Using a single client for connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=30.0
        )

    async def close(self):
        await self.client.aclose()

    async def start_test_execution(self, run_test_id: str) -> Dict[str, Any]:
        """
        POST /simulate/run-tests/{run_test_id}/chat-execute/
        Starts a test execution and returns the execution ID.
        Note: The backend uses scenarios associated with the run_test_id.
        """
        url = f"/simulate/run-tests/{run_test_id}/chat-execute/"
        # Empty body - backend uses scenarios from run_test
        response = await self.client.post(url, json={})
        response.raise_for_status()
        return response.json()

    async def fetch_execution_batch(
        self, 
        run_test_id: str, 
        test_execution_id: str
    ) -> Dict[str, Any]:
        """
        GET /simulate/run-tests/{run_test_id}/chat-execute/?test_execution_id=...
        Fetches a batch of call execution IDs.
        """
        url = f"/simulate/run-tests/{run_test_id}/chat-execute/"
        params = {"test_execution_id": test_execution_id}
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def send_chat_message(
        self, 
        call_execution_id: str, 
        messages: List[Dict[str, str]] = None,
        metrics: Dict[str, float | int] = None,
        initiate_chat: bool = False
    ) -> Dict[str, Any]:
        """
        POST /simulate/call-executions/{call_execution_id}/chat/send-message/
        Sends a message to a chat execution.
        """
        url = f"/simulate/call-executions/{call_execution_id}/chat/send-message/"
        
        payload = {
            "messages": messages,
            "metrics": metrics,
            "initiate_chat": initiate_chat
        }
        # Filter None values (but keep False for booleans if needed, though backend defaults to False)
        # We explicitly keep initiate_chat if it's True
        payload = {k: v for k, v in payload.items() if v is not None}
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

