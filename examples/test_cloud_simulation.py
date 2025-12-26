import asyncio
import os
from typing import Union, List, Dict, Optional
from datetime import datetime, timedelta
from agents import Agent, Runner, SQLiteSession, function_tool
from fi.simulate import TestRunner, AgentWrapper, AgentInput, AgentResponse

# Ensure you have set these environment variables:
# export FI_API_KEY="your-api-key"
# export FI_SECRET_KEY="your-secret-key"
# export FI_BASE_URL="http://localhost:8000"  # Or your dev/prod URL
# export OPENAI_API_KEY="your-openai-key"

# Install OpenAI Agents library:
#   pip install openai-agents
#   # Or with uv: uv add openai-agents


# ============================================================================
# DELIVERY SUPPORT TOOLS
# ============================================================================

# Mock database for demonstration purposes
_orders_db = {
    "ORD-12345": {
        "order_id": "ORD-12345",
        "customer_id": "CUST-001",
        "customer_name": "John Doe",
        "items": ["Product A", "Product B"],
        "status": "in_transit",
        "estimated_delivery": "2024-01-15",
        "tracking_number": "TRACK-789",
        "address": "123 Main St, City, State 12345",
        "created_at": "2024-01-10"
    },
    "ORD-67890": {
        "order_id": "ORD-67890",
        "customer_id": "CUST-002",
        "customer_name": "Jane Smith",
        "items": ["Product C"],
        "status": "delivered",
        "estimated_delivery": "2024-01-12",
        "tracking_number": "TRACK-456",
        "address": "456 Oak Ave, City, State 67890",
        "delivered_at": "2024-01-12",
        "created_at": "2024-01-08"
    },
    "ORD-11111": {
        "order_id": "ORD-11111",
        "customer_id": "CUST-001",
        "customer_name": "John Doe",
        "items": ["Product D"],
        "status": "processing",
        "estimated_delivery": "2024-01-20",
        "tracking_number": None,
        "address": "123 Main St, City, State 12345",
        "created_at": "2024-01-13"
    }
}

_customers_db = {
    "CUST-001": {
        "customer_id": "CUST-001",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+1-555-0101",
        "order_history": ["ORD-12345", "ORD-11111"]
    },
    "CUST-002": {
        "customer_id": "CUST-002",
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "phone": "+1-555-0202",
        "order_history": ["ORD-67890"]
    }
}


@function_tool
def track_order(order_id: str) -> Dict:
    """
    Track an order by its order ID. Returns detailed information about the order including
    current status, estimated delivery date, tracking number, and delivery address.
    
    Args:
        order_id: The order ID to track (e.g., "ORD-12345")
    
    Returns:
        A dictionary containing order details including status, tracking info, and delivery address
    """
    print(f"🔧 [TOOL CALL] track_order(order_id='{order_id}')")
    order = _orders_db.get(order_id.upper())
    if not order:
        result = {
            "error": f"Order {order_id} not found",
            "order_id": order_id
        }
        print(f"   ❌ Result: {result}")
        return result
    
    result = {
        "order_id": order["order_id"],
        "status": order["status"],
        "estimated_delivery": order.get("estimated_delivery"),
        "tracking_number": order.get("tracking_number"),
        "items": order["items"],
        "delivery_address": order["address"],
        "customer_name": order["customer_name"]
    }
    print(f"   ✅ Result: Order found - Status: {order['status']}")
    return result


@function_tool
def get_delivery_status(order_id: str) -> Dict:
    """
    Get the current delivery status of an order. Provides real-time status updates
    including whether the order is processing, in transit, out for delivery, or delivered.
    
    Args:
        order_id: The order ID to check status for
    
    Returns:
        A dictionary with the current delivery status and relevant timestamps
    """
    print(f"🔧 [TOOL CALL] get_delivery_status(order_id='{order_id}')")
    order = _orders_db.get(order_id.upper())
    if not order:
        result = {
            "error": f"Order {order_id} not found",
            "order_id": order_id
        }
        print(f"   ❌ Result: {result}")
        return result
    
    status_messages = {
        "processing": "Your order is being prepared for shipment",
        "in_transit": "Your order is on the way",
        "out_for_delivery": "Your order is out for delivery today",
        "delivered": "Your order has been delivered"
    }
    
    result = {
        "order_id": order["order_id"],
        "status": order["status"],
        "status_message": status_messages.get(order["status"], "Unknown status"),
        "estimated_delivery": order.get("estimated_delivery")
    }
    
    if order["status"] == "delivered" and "delivered_at" in order:
        result["delivered_at"] = order["delivered_at"]
    
    if order.get("tracking_number"):
        result["tracking_number"] = order["tracking_number"]
    
    print(f"   ✅ Result: Status = {order['status']}")
    return result


@function_tool
def update_delivery_address(order_id: str, new_address: str) -> Dict:
    """
    Update the delivery address for an order. Can only be updated if the order
    hasn't been shipped yet.
    
    Args:
        order_id: The order ID to update
        new_address: The new delivery address
    
    Returns:
        A dictionary indicating success or failure of the address update
    """
    print(f"🔧 [TOOL CALL] update_delivery_address(order_id='{order_id}', new_address='{new_address}')")
    order = _orders_db.get(order_id.upper())
    if not order:
        result = {
            "success": False,
            "error": f"Order {order_id} not found"
        }
        print(f"   ❌ Result: {result}")
        return result
    
    if order["status"] in ["in_transit", "out_for_delivery", "delivered"]:
        result = {
            "success": False,
            "error": f"Cannot update address. Order is already {order['status']}",
            "order_id": order_id,
            "current_status": order["status"]
        }
        print(f"   ❌ Result: Cannot update - Order is {order['status']}")
        return result
    
    # Update the address
    _orders_db[order_id.upper()]["address"] = new_address
    
    result = {
        "success": True,
        "order_id": order_id,
        "new_address": new_address,
        "message": "Delivery address updated successfully"
    }
    print(f"   ✅ Result: Address updated successfully")
    return result


@function_tool
def cancel_order(order_id: str, reason: Optional[str] = None) -> Dict:
    """
    Cancel an order. Orders can only be cancelled if they haven't been shipped yet.
    
    Args:
        order_id: The order ID to cancel
        reason: Optional reason for cancellation
    
    Returns:
        A dictionary indicating success or failure of the cancellation
    """
    print(f"🔧 [TOOL CALL] cancel_order(order_id='{order_id}', reason='{reason}')")
    order = _orders_db.get(order_id.upper())
    if not order:
        result = {
            "success": False,
            "error": f"Order {order_id} not found"
        }
        print(f"   ❌ Result: {result}")
        return result
    
    if order["status"] in ["in_transit", "out_for_delivery", "delivered"]:
        result = {
            "success": False,
            "error": f"Cannot cancel order. Order is already {order['status']}",
            "order_id": order_id,
            "current_status": order["status"]
        }
        print(f"   ❌ Result: Cannot cancel - Order is {order['status']}")
        return result
    
    # Cancel the order
    _orders_db[order_id.upper()]["status"] = "cancelled"
    if reason:
        _orders_db[order_id.upper()]["cancellation_reason"] = reason
    
    result = {
        "success": True,
        "order_id": order_id,
        "message": "Order cancelled successfully",
        "refund_expected": "Refund will be processed within 5-7 business days"
    }
    print(f"   ✅ Result: Order cancelled successfully")
    return result


@function_tool
def get_order_history(customer_id: Optional[str] = None, customer_email: Optional[str] = None) -> Dict:
    """
    Get order history for a customer. Can search by customer ID or email address.
    Returns a list of all orders associated with the customer.
    
    Args:
        customer_id: The customer ID to look up (e.g., "CUST-001")
        customer_email: The customer email to look up (alternative to customer_id)
    
    Returns:
        A dictionary containing customer information and their order history
    """
    print(f"🔧 [TOOL CALL] get_order_history(customer_id='{customer_id}', customer_email='{customer_email}')")
    customer = None
    
    if customer_id:
        customer = _customers_db.get(customer_id.upper())
    elif customer_email:
        # Find customer by email
        for cust in _customers_db.values():
            if cust["email"].lower() == customer_email.lower():
                customer = cust
                break
    
    if not customer:
        result = {
            "error": "Customer not found",
            "customer_id": customer_id,
            "customer_email": customer_email
        }
        print(f"   ❌ Result: {result}")
        return result
    
    # Get all orders for this customer
    orders = []
    for order_id in customer.get("order_history", []):
        order = _orders_db.get(order_id)
        if order:
            orders.append({
                "order_id": order["order_id"],
                "status": order["status"],
                "items": order["items"],
                "created_at": order["created_at"],
                "estimated_delivery": order.get("estimated_delivery")
            })
    
    result = {
        "customer_id": customer["customer_id"],
        "customer_name": customer["name"],
        "total_orders": len(orders),
        "orders": orders
    }
    print(f"   ✅ Result: Found {len(orders)} orders for {customer['name']}")
    return result


@function_tool
def get_tracking_updates(tracking_number: str) -> Dict:
    """
    Get detailed tracking updates for a shipment using the tracking number.
    Provides location updates and estimated delivery time.
    
    Args:
        tracking_number: The tracking number (e.g., "TRACK-789")
    
    Returns:
        A dictionary with tracking updates and current location
    """
    print(f"🔧 [TOOL CALL] get_tracking_updates(tracking_number='{tracking_number}')")
    # Find order by tracking number
    order = None
    for ord in _orders_db.values():
        if ord.get("tracking_number") == tracking_number.upper():
            order = ord
            break
    
    if not order:
        result = {
            "error": f"Tracking number {tracking_number} not found",
            "tracking_number": tracking_number
        }
        print(f"   ❌ Result: {result}")
        return result
    
    # Generate mock tracking updates based on status
    updates = []
    if order["status"] == "processing":
        updates.append({
            "timestamp": order["created_at"],
            "location": "Warehouse",
            "status": "Order received and being processed"
        })
    elif order["status"] == "in_transit":
        updates.extend([
            {
                "timestamp": order["created_at"],
                "location": "Warehouse",
                "status": "Order shipped"
            },
            {
                "timestamp": str(datetime.now().date() - timedelta(days=2)),
                "location": "Distribution Center",
                "status": "In transit to local facility"
            }
        ])
    elif order["status"] == "delivered":
        updates.extend([
            {
                "timestamp": order["created_at"],
                "location": "Warehouse",
                "status": "Order shipped"
            },
            {
                "timestamp": order.get("delivered_at", order["estimated_delivery"]),
                "location": order["address"],
                "status": "Delivered"
            }
        ])
    
    result = {
        "tracking_number": tracking_number.upper(),
        "order_id": order["order_id"],
        "current_status": order["status"],
        "updates": updates,
        "estimated_delivery": order.get("estimated_delivery")
    }
    print(f"   ✅ Result: Found {len(updates)} tracking updates, Status = {order['status']}")
    return result


class OpenAIAgentsWrapper(AgentWrapper):
    """
    Custom wrapper for OpenAI Agents library (https://github.com/openai/openai-agents-python).
    This allows you to use OpenAI Agents framework with the Simulate SDK.
    """
    
    def __init__(self, agent: Agent, use_session: bool = True):
        """
        Args:
            agent: An OpenAI Agents Agent instance
            use_session: If True, uses SQLiteSession to maintain conversation history.
                        If False, passes only the last message as input.
        """
        self.agent = agent
        self.use_session = use_session
        # Store sessions per thread_id to maintain separate conversation histories
        self._sessions = {}
    
    async def call(self, input: AgentInput) -> Union[str, AgentResponse]:
        """
        Process the input using OpenAI Agents Runner and return the response.
        """
        # Extract the last user message as input for the agent
        # OpenAI Agents Runner.run() expects a string input
        if input.new_message:
            user_input = input.new_message.get("content", "")
        elif input.messages:
            # Fallback: get the last user message from history
            for msg in reversed(input.messages):
                if msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break
            else:
                user_input = ""
        else:
            user_input = ""
        
        if not user_input:
            return "I didn't receive a message to respond to."
        
        # Use session-based approach for maintaining conversation context
        if self.use_session:
            # Create or retrieve session for this thread
            session_id = input.thread_id or input.execution_id or "default"
            if session_id not in self._sessions:
                # Use in-memory SQLite session (or file-based: SQLiteSession(session_id, "conversations.db"))
                self._sessions[session_id] = SQLiteSession(session_id)
            
            session = self._sessions[session_id]
            
            # Run the agent with session (automatically maintains history)
            result = await Runner.run(
                self.agent,
                user_input,
                session=session
            )
        else:
            # Stateless mode: just pass the current input
            # Note: This doesn't use conversation history, but is simpler
            result = await Runner.run(
                self.agent,
                user_input
            )
        
        # Extract the final output and tool calls from the result
        final_output = result.final_output if hasattr(result, 'final_output') else str(result)
        
        # Extract tool calls and tool responses from result.items (the correct way per OpenAI Agents docs)
        tool_calls = []
        tool_responses = []
        try:
            from openai.agents.types import FunctionCallItem, ComputerUseCallItem
            
            if hasattr(result, 'items') and result.items:
                for item in result.items:
                    if isinstance(item, (FunctionCallItem, ComputerUseCallItem)):
                        # Extract tool_call from the item
                        tool_call = item.tool_call
                        if tool_call:
                            # Convert to OpenAI format
                            tool_calls.append({
                                "id": getattr(tool_call, 'id', None),
                                "type": "function",
                                "function": {
                                    "name": getattr(tool_call.function, 'name', None) if hasattr(tool_call, 'function') else None,
                                    "arguments": getattr(tool_call.function, 'arguments', None) if hasattr(tool_call, 'function') else None
                                }
                            })
                        
                        # Extract tool response/result if available
                        if hasattr(item, 'result') or hasattr(item, 'output'):
                            tool_result = getattr(item, 'result', None) or getattr(item, 'output', None)
                            if tool_result and tool_call:
                                # Create tool role message
                                import json
                                tool_responses.append({
                                    "role": "tool",
                                    "tool_call_id": getattr(tool_call, 'id', None),
                                    "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                                })
        except ImportError:
            # Fallback if types aren't available
            if hasattr(result, 'items') and result.items:
                for item in result.items:
                    # Check if item has tool_call attribute
                    if hasattr(item, 'tool_call') and item.tool_call:
                        tool_call = item.tool_call
                        tool_calls.append({
                            "id": getattr(tool_call, 'id', None),
                            "type": "function",
                            "function": {
                                "name": getattr(tool_call.function, 'name', None) if hasattr(tool_call, 'function') else None,
                                "arguments": getattr(tool_call.function, 'arguments', None) if hasattr(tool_call, 'function') else None
                            }
                        })
                        
                        # Extract tool response if available
                        if hasattr(item, 'result') or hasattr(item, 'output'):
                            tool_result = getattr(item, 'result', None) or getattr(item, 'output', None)
                            if tool_result:
                                import json
                                tool_responses.append({
                                    "role": "tool",
                                    "tool_call_id": getattr(tool_call, 'id', None),
                                    "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                                })
        
        # Return AgentResponse if we have tool calls or responses, otherwise just the string
        if tool_calls or tool_responses:
            return AgentResponse(
                content=final_output,
                tool_calls=tool_calls if tool_calls else None,
                tool_responses=tool_responses if tool_responses else None
            )
        else:
            return final_output

async def main():
    # 1. Create an OpenAI Agents Agent with tools
    # This is a delivery/customer service agent example with real tools
    delivery_agent = Agent(
        name="Delivery Support Agent",
        instructions="""You are a helpful delivery support agent. Your role is to:
- Help customers track their orders using the track_order tool
- Answer questions about delivery status using get_delivery_status
- Update delivery addresses when requested (if order hasn't shipped)
- Cancel orders when requested (if order hasn't shipped)
- Look up order history for customers
- Get tracking updates using tracking numbers

You have access to several tools to help customers:
- track_order: Get full order details by order ID
- get_delivery_status: Check current delivery status
- update_delivery_address: Update address for unshipped orders
- cancel_order: Cancel orders that haven't shipped
- get_order_history: Look up all orders for a customer
- get_tracking_updates: Get detailed tracking information

Always use the appropriate tool when a customer asks about their order. Be proactive
in using tools to get accurate information rather than guessing.

Be concise, helpful, and empathetic. If you don't have specific order information, 
guide the customer on how to find it or contact support.""",
        model="gpt-4o-mini",  # You can specify the model here
        tools=[
            track_order,
            get_delivery_status,
            update_delivery_address,
            cancel_order,
            get_order_history,
            get_tracking_updates
        ]
    )
    
    # 2. Wrap it for the Simulate SDK
    # use_session=True maintains conversation history across turns
    agent_wrapper = OpenAIAgentsWrapper(
        agent=delivery_agent,
        use_session=True  # Set to False for stateless mode
    )

    # 3. Configure the Test Runner
    runner = TestRunner(
        # Credentials can be passed here or via env vars
        # api_key=os.environ.get("FI_API_KEY"), 
        # secret_key=os.environ.get("FI_SECRET_KEY")
    )

    print("🚀 Starting Cloud Simulation Test with OpenAI Agents...")
    print("📦 Using Delivery Support Agent")
    print("🛠️  Tools available:")
    print("   - track_order")
    print("   - get_delivery_status")
    print("   - update_delivery_address")
    print("   - cancel_order")
    print("   - get_order_history")
    print("   - get_tracking_updates")

    # 4. Run the simulation
    # 'run_id' corresponds to the 'run_test_id' from the Future AGI platform
    # Replace with a valid ID from your database/platform
    RUN_ID = "0ea34d43-ded5-4087-82e9-61de84fc74fe" 

    try:
        report = await runner.run_test(
            run_id=RUN_ID,
            agent_callback=agent_wrapper,
            concurrency=1,  # Number of parallel conversations to handle
        )
        
        print(f"\n✅ Simulation finished!")
        print(f"Total results processed: {len(report.results)}")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

