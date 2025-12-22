import asyncio
import os
import contextvars
import logging
from typing import Optional, Callable, Dict, Any, List

from ...agent.wrapper import AgentWrapper, AgentInput, AgentResponse
from ..models import TestReport
from .base import BaseEngine
from ...utils.routes import APIRoutes

# Context variable to track the current execution ID for future tool mocking
current_execution_id = contextvars.ContextVar("current_execution_id", default=None)

logger = logging.getLogger(__name__)

class CloudEngine(BaseEngine):
    """
    Execution engine that connects to the Future AGI backend to orchestrate simulations.
    It acts as a bridge between the cloud-hosted simulator and the user's local agent.
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, api_url: Optional[str] = None, timeout: float = 120.0):
        """
        Args:
            api_key: API key for authentication
            secret_key: Secret key for authentication
            api_url: Base URL of the backend API
            timeout: Request timeout in seconds (default: 120s for LLM operations)
        """
        self.api_key = api_key or os.environ.get("FI_API_KEY")
        self.secret_key = secret_key or os.environ.get("FI_SECRET_KEY")
        self.api_url = api_url or os.environ.get("FI_BASE_URL") or "https://api.futureagi.com"
        self.timeout = timeout
        
        if not self.api_key or not self.secret_key:
            logger.warning("FI_API_KEY or FI_SECRET_KEY not provided. CloudEngine will not function correctly.")
            
        self.api = None

    async def run(
        self,
        run_id: Optional[str] = None,
        agent_callback: Optional[Callable | AgentWrapper] = None,
        concurrency: int = 5,
        **kwargs
    ) -> TestReport:
        """
        Connects to the cloud run, receives user inputs, calls the agent_callback,
        and sends responses back.
        """
        if not run_id:
            raise ValueError("CloudEngine requires a 'run_id' (run_test_id).")
        
        if not agent_callback:
            raise ValueError("CloudEngine requires an 'agent_callback' (function or AgentWrapper).")

        self.api = APIRoutes(self.api_key, self.secret_key, self.api_url, timeout=self.timeout)
        wrapper = self._normalize_callback(agent_callback)
        queue = asyncio.Queue()

        print(f"Starting Simulation for Run ID: {run_id}")
        
        try:
            # 1. Start the Run (Create TestExecution)
            start_resp = await self.api.start_test_execution(run_test_id=run_id)
            result = start_resp.get("result", {})
            # Handle both camelCase and snake_case response formats
            test_execution_id = result.get("executionId") or result.get("execution_id")
            
            if not test_execution_id:
                raise ValueError(f"Failed to start test execution. Response: {start_resp}")
                
            print(f"✓ Test Execution Started: {test_execution_id}")

            # 2. Start Producer and Consumers
            producer_task = asyncio.create_task(
                self._producer_loop(run_id, test_execution_id, queue)
            )
            
            consumers = [
                asyncio.create_task(self._consumer_loop(queue, wrapper))
                for _ in range(concurrency)
            ]
            
            # Wait for producer to finish fetching all batches
            await producer_task
            
            # Wait for queue to drain (all consumers process remaining items)
            await queue.join()
            
            # Cancel consumers
            for c in consumers:
                c.cancel()
                
            print("✅ Cloud Simulation Completed.")
            
        except Exception as e:
            logger.exception(f"Cloud simulation failed: {e}")
            raise
        finally:
            if self.api:
                await self.api.close()
        
        # Return empty report for now as backend handles metrics
        return TestReport(results=[])

    async def _producer_loop(self, run_test_id: str, test_execution_id: str, queue: asyncio.Queue):
        """
        Polls the backend for batches of call execution IDs and puts them in the queue.
        """
        has_more = True
        
        while has_more:
            try:
                print("🔄 Fetching batch of scenarios...")
                resp = await self.api.fetch_execution_batch(run_test_id, test_execution_id)
                
                result = resp.get("result", {})
                # Handle both camelCase and snake_case response formats
                call_ids = result.get("callExecutionIds") or result.get("call_execution_ids", [])
                has_more = result.get("hasMore") if "hasMore" in result else result.get("has_more", False)
                
                if not call_ids:
                    if has_more:
                        print("⚠️ Received empty batch but hasMore is true. Waiting...")
                        await asyncio.sleep(2)
                        continue
                    else:
                        break
                
                print(f"📥 Received batch: {len(call_ids)} calls")
                for cid in call_ids:
                    await queue.put(cid)
                    
            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                # Simple retry logic or break? For now, break to avoid infinite loop
                break

    async def _consumer_loop(self, queue: asyncio.Queue, wrapper: AgentWrapper):
        """
        Worker that pulls execution IDs from the queue and runs the conversation.
        """
        while True:
            try:
                execution_id = await queue.get()
                await self._handle_single_execution(execution_id, wrapper)
                queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = str(e) or f"{type(e).__name__}: {repr(e)}"
                logger.error(f"Error in consumer: {error_msg}", exc_info=True)
                print(f"❌ Consumer error: {error_msg}")
                queue.task_done() # Mark done even if failed so join() works

    async def _handle_single_execution(self, call_execution_id: str, wrapper: AgentWrapper):
        """
        Runs the conversation loop for a single call execution.
        """
        token = current_execution_id.set(call_execution_id)
        try:
            print(f"▶️ Processing Call: {call_execution_id}")
            
            # Step 1: Initiate chat (POST with initiate_chat=True)
            init_resp = await self.api.send_chat_message(
                call_execution_id=call_execution_id,
                initiate_chat=True
            )
            result = init_resp.get("result", {})
            
            if not result:
                logger.error(f"Failed to initiate chat for {call_execution_id}")
                return
            
            # Extract first message(s) from response
            # Note: message_history is a list of ChatMessage objects (dicts)
            message_history = result.get("message_history") or result.get("messageHistory", [])
            
            if not message_history:
                # Fallback to output_message if history is empty
                output_msg = result.get("output_message") or result.get("outputMessage")
                if output_msg:
                    # Ensure it's a list
                    if isinstance(output_msg, list):
                        message_history = output_msg
                    else:
                        message_history = [output_msg]
            
            if not message_history:
                logger.warning(f"No initial message received for {call_execution_id}")
                return
            
            # Build conversation history for SDK format
            # Convert backend "assistant" → SDK "user" (simulator messages)
            conversation_history = []
            for msg in message_history:
                backend_role = msg.get("role", "user")
                
                # Filter out system and tool messages from backend (simulator artifacts)
                if backend_role in ["system", "tool"]:
                    continue
                
                # Filter out empty messages (often tool calls without output text yet)
                content = msg.get("content", "")
                if not content and backend_role == "assistant":
                    continue

                # Backend sends simulator messages as "assistant", convert to "user" for SDK
                sdk_role = "user" if backend_role == "assistant" else backend_role
                conversation_history.append({
                    "role": sdk_role,
                    "content": content
                })
            
            # Step 2: Conversation loop
            max_turns = 50  # Safety limit
            turn_count = 0
            
            while turn_count < max_turns:
                # Check if chat ended based on last response
                chat_ended = result.get("chat_ended") or result.get("chatEnded", False)
                if chat_ended:
                    break

                # Get the last message (should be from simulator/user to reply to)
                if not conversation_history:
                    break
                    
                last_msg = conversation_history[-1]
                
                # Prepare AgentInput for user's wrapper
                agent_input = AgentInput(
                    thread_id=call_execution_id,
                    messages=conversation_history,
                    new_message=last_msg,
                    execution_id=call_execution_id
                )
                
                # Call user's agent and measure latency
                import time
                start_time = time.time()
                try:
                    agent_response = await wrapper.call(agent_input)
                except Exception as e:
                    error_msg = str(e) or f"{type(e).__name__}: {repr(e)}"
                    last_msg_content = agent_input.new_message.get('content', '') if agent_input.new_message else 'N/A'
                    logger.error(f"Agent call failed for {call_execution_id}: {error_msg}", exc_info=True)
                    print(f"❌ Agent call failed for {call_execution_id}: {error_msg}")
                    if last_msg_content != 'N/A':
                        print(f"   Last message: {last_msg_content[:100]}...")
                    break
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Normalize response and extract tool_calls
                response_content = ""
                tool_calls = None
                
                if isinstance(agent_response, AgentResponse):
                    response_content = agent_response.content
                    tool_calls = agent_response.tool_calls
                else:
                    response_content = str(agent_response)
                
                # Add agent response to history (with tool_calls if present)
                assistant_msg = {
                    "role": "assistant",
                    "content": response_content
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                conversation_history.append(assistant_msg)
                
                # Step 3: Send agent response to backend and get next message
                # Only send the new message (Agent response) to avoid duplication as backend appends
                # SDK "assistant" (agent) → backend "user"
                last_msg = conversation_history[-1]
                
                api_message = {
                    "role": "user",
                    "content": last_msg["content"]
                }
                # Include tool_calls if present
                if "tool_calls" in last_msg:
                    api_message["tool_calls"] = last_msg["tool_calls"]
                
                api_messages = [api_message]
                
                metrics = {"latency": latency_ms}
                
                # Send
                turn_resp = await self.api.send_chat_message(
                    call_execution_id=call_execution_id,
                    messages=api_messages,
                    metrics=metrics,
                    initiate_chat=False
                )
                
                result = turn_resp.get("result", {})
                if not result:
                    logger.warning(f"No response from backend for {call_execution_id}")
                    break
                
                # Update conversation history from backend response

                new_history_data = result.get("message_history") or result.get("messageHistory", [])
                
                if new_history_data:
                    # Convert backend "assistant" → SDK "user" (simulator messages)
                    conversation_history = []
                    for msg in new_history_data:
                        backend_role = msg.get("role", "user")
                        
                        # Filter out system and tool messages
                        if backend_role in ["system", "tool"]:
                            continue

                        # Filter out empty messages
                        content = msg.get("content", "")
                        if not content and backend_role == "assistant":
                            continue

                        sdk_role = "user" if backend_role == "assistant" else backend_role
                        conversation_history.append({
                            "role": sdk_role,
                            "content": content
                        })
                else:
                    # Fallback: append output_message if history missing
                    output_msgs = result.get("output_message") or result.get("outputMessage")
                    if output_msgs:
                        if isinstance(output_msgs, list):
                            for om in output_msgs:
                                backend_role = om.get("role", "user")
                                if backend_role in ["system", "tool"]:
                                    continue
                                
                                content = om.get("content", "")
                                if not content and backend_role == "assistant":
                                    continue

                                sdk_role = "user" if backend_role == "assistant" else backend_role
                                conversation_history.append({
                                    "role": sdk_role,
                                    "content": content
                                })
                        else:
                            backend_role = output_msgs.get("role", "user")
                            if backend_role not in ["system", "tool"]:
                                content = output_msgs.get("content", "")
                                if content or backend_role != "assistant":
                                    sdk_role = "user" if backend_role == "assistant" else backend_role
                                    conversation_history.append({
                                        "role": sdk_role,
                                        "content": content
                                    })
                
                turn_count += 1
            
            print(f"✓ Call Finished: {call_execution_id} ({turn_count} turns)")
            
        except Exception as e:
            # Get detailed error message
            error_msg = str(e)
            if not error_msg:
                error_msg = f"{type(e).__name__}: {repr(e)}"
            
            # Log to both logger and console
            logger.error(f"Call execution {call_execution_id} failed: {error_msg}", exc_info=True)
            print(f"❌ Call execution {call_execution_id} failed: {error_msg}")
        finally:
            current_execution_id.reset(token)

    def _normalize_callback(self, callback: Callable | AgentWrapper) -> AgentWrapper:
        """Ensures we have a AgentWrapper instance."""
        if isinstance(callback, AgentWrapper):
            return callback
        
        # If it's a function, wrap it
        class FunctionalWrapper(AgentWrapper):
            def __init__(self, func):
                self.func = func
            
            async def call(self, input: AgentInput) -> str | AgentResponse:
                # Support both async and sync functions
                if asyncio.iscoroutinefunction(self.func):
                    return await self.func(input)
                return self.func(input)
                
        return FunctionalWrapper(callback)
