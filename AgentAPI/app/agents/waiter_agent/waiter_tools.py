import aiohttp
import json
import uuid
import hashlib
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from config import config as app_config
from pathlib import Path
import yaml

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    with open(yaml_path, 'r') as f:
        descriptions = yaml.safe_load(f)
    return descriptions.get(tool_name, "")

# =============================================================================
# MENU MANAGEMENT TOOLS
# =============================================================================

class MenuQuery(BaseModel):
    category: Optional[str] = Field(default=None, description="Menu category: appetizers, mains, desserts, beverages")
    dietary_filters: Optional[List[str]] = Field(default=None, description="Dietary restrictions: vegetarian, vegan, gluten-free, etc.")
    price_range: Optional[str] = Field(default=None, description="Price range: low, medium, high")

@tool(
    name_or_callable="browse_menu_api",
    description=get_tool_description('browse_menu_api'),
    args_schema=MenuQuery,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def browse_menu_api(category: Optional[str] = None, dietary_filters: Optional[List[str]] = None, price_range: Optional[str] = None, config: RunnableConfig = None) -> str:
    """Fetch menu items from restaurant system with filtering options"""
    
    menu_data = {
        "appetizers": [
            {"id": "app_001", "name": "Garlic Bread", "price": 8.99, "description": "Fresh baked with herbs", "dietary": ["vegetarian"]},
            {"id": "app_002", "name": "Caesar Salad", "price": 12.99, "description": "Crisp romaine with parmesan", "dietary": ["vegetarian"]},
            {"id": "app_003", "name": "Calamari Rings", "price": 14.99, "description": "Crispy fried squid with marinara", "dietary": []},
            {"id": "app_004", "name": "Hummus Platter", "price": 10.99, "description": "With fresh vegetables and pita", "dietary": ["vegan", "vegetarian"]}
        ],
        "mains": [
            {"id": "main_001", "name": "Grilled Salmon", "price": 24.99, "description": "Atlantic salmon with vegetables", "dietary": ["gluten-free"]},
            {"id": "main_002", "name": "Ribeye Steak", "price": 29.99, "description": "12oz ribeye with choice of sides", "dietary": ["gluten-free"]},
            {"id": "main_003", "name": "Vegetable Curry", "price": 18.99, "description": "Coconut curry with jasmine rice", "dietary": ["vegan", "vegetarian"]},
            {"id": "main_004", "name": "Chicken Parmesan", "price": 22.99, "description": "Breaded chicken with pasta", "dietary": []}
        ],
        "desserts": [
            {"id": "des_001", "name": "Chocolate Cake", "price": 7.99, "description": "Rich chocolate layer cake", "dietary": ["vegetarian"]},
            {"id": "des_002", "name": "Fruit Sorbet", "price": 6.99, "description": "Seasonal fruit sorbet", "dietary": ["vegan", "vegetarian", "gluten-free"]}
        ],
        "beverages": [
            {"id": "bev_001", "name": "House Wine", "price": 8.99, "description": "Red or white selection", "dietary": ["vegetarian"]},
            {"id": "bev_002", "name": "Craft Beer", "price": 6.99, "description": "Local brewery selection", "dietary": []},
            {"id": "bev_003", "name": "Fresh Juice", "price": 4.99, "description": "Orange, apple, or cranberry", "dietary": ["vegan", "vegetarian"]}
        ]
    }
    
    # Filter by category
    if category and category in menu_data:
        items = menu_data[category]
    else:
        items = []
        for cat_items in menu_data.values():
            items.extend(cat_items)
    
    # Filter by dietary preferences
    if dietary_filters:
        filtered_items = []
        for item in items:
            if any(diet in item.get("dietary", []) for diet in dietary_filters):
                filtered_items.append(item)
        items = filtered_items
    
    # Filter by price range
    if price_range:
        if price_range == "low":
            items = [item for item in items if item["price"] < 15.00]
        elif price_range == "medium":
            items = [item for item in items if 15.00 <= item["price"] <= 25.00]
        elif price_range == "high":
            items = [item for item in items if item["price"] > 25.00]
    
    return str({"items": items, "total_count": len(items)})


class AvailabilityCheck(BaseModel):
    item_id: str = Field(description="Menu item ID to check availability")
    quantity: int = Field(default=1, description="Quantity needed")
    special_requests: Optional[str] = Field(default=None, description="Special preparation requests")

@tool(
    name_or_callable="check_item_availability_api",
    description=get_tool_description('check_item_availability_api'),
    args_schema=AvailabilityCheck,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def check_item_availability_api(item_id: str, quantity: int = 1, special_requests: Optional[str] = None, config: RunnableConfig = None) -> str:
    """Check if menu item is available with current stock and kitchen capacity"""
    
    availability_data = {
        "available": True,
        "stock_level": "high",
        "estimated_prep_time": "15-20 minutes",
        "kitchen_capacity": "normal",
        "special_notes": None
    }
    
    # Simulate some unavailable items
    if item_id in ["main_002"]:  # Ribeye sometimes unavailable
        availability_data.update({
            "available": False,
            "reason": "Temporarily out of stock",
            "alternative_suggestions": ["main_001", "main_004"]
        })
    
    if special_requests:
        availability_data["special_request_feasible"] = True
        availability_data["additional_prep_time"] = "5-10 minutes"

    return str(availability_data)


# =============================================================================
# ORDER MANAGEMENT TOOLS
# =============================================================================

class OrderSubmission(BaseModel):
    order_items: List[Dict[str, Any]] = Field(description="List of ordered items with details")
    table_number: str = Field(description="Table number for the order")
    customer_name: Optional[str] = Field(default=None, description="Customer name for the order")
    special_instructions: Optional[str] = Field(default=None, description="Special cooking or service instructions")
    dietary_restrictions: Optional[List[str]] = Field(default=None, description="Customer dietary restrictions")

@tool(
    name_or_callable="submit_order_to_kitchen_api",
    description=get_tool_description('submit_order_to_kitchen_api'),
    args_schema=OrderSubmission
)
async def submit_order_to_kitchen_api(order_items: List[Dict[str, Any]], table_number: str, customer_name: Optional[str] = None, 
                                    special_instructions: Optional[str] = None, dietary_restrictions: Optional[List[str]] = None, 
                                    config: RunnableConfig = None) -> str:
    """Submit order to kitchen system with all details"""
    
    order_id = f"ORD_{uuid.uuid4().hex[:8].upper()}"
    
    # Calculate total and prep time
    total_price = sum(item.get("price", 0) * item.get("quantity", 1) for item in order_items)
    estimated_time = max(20, len(order_items) * 8)  # Base time + per item
    
    kitchen_response = {
        "order_id": order_id,
        "status": "received",
        "table_number": table_number,
        "customer_name": customer_name,
        "total_items": len(order_items),
        "total_price": round(total_price, 2),
        "estimated_completion": f"{estimated_time}-{estimated_time + 10} minutes",
        "kitchen_notes": "Order received and queued for preparation",
        "priority": "normal",
        "dietary_alerts": dietary_restrictions or []
    }
    
    if special_instructions:
        kitchen_response["special_instructions"] = special_instructions
        kitchen_response["kitchen_notes"] += " - Special instructions noted"
    
    return str(kitchen_response)


class OrderStatusQuery(BaseModel):
    order_id: str = Field(description="Order ID to check status")

@tool(
    name_or_callable="check_order_status_api",
    description=get_tool_description('check_order_status_api'),
    args_schema=OrderStatusQuery,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def check_order_status_api(order_id: str, config: RunnableConfig = None) -> str:
    """Check order status from kitchen system"""
    
    # Mock status progression
    import hashlib
    status_hash = int(hashlib.md5(order_id.encode()).hexdigest()[:8], 16)
    progress_stage = status_hash % 5
    
    status_map = {
        0: {"status": "received", "progress": "10%", "eta": "25-30 minutes"},
        1: {"status": "preparing", "progress": "30%", "eta": "20-25 minutes"},
        2: {"status": "cooking", "progress": "60%", "eta": "10-15 minutes"},
        3: {"status": "finishing", "progress": "85%", "eta": "5-8 minutes"},
        4: {"status": "ready", "progress": "100%", "eta": "Ready for pickup"}
    }
    
    current_status = status_map[progress_stage]
    
    order_status = {
        "order_id": order_id,
        "status": current_status["status"],
        "progress": current_status["progress"],
        "estimated_completion": current_status["eta"],
        "items_ready": [],
        "kitchen_notes": f"Order is currently {current_status['status']}",
        "last_updated": "2 minutes ago"
    }
    
    if current_status["status"] == "ready":
        order_status["pickup_ready"] = True
        order_status["kitchen_notes"] = "Order ready for service"
    
    return str(order_status)


class OrderModification(BaseModel):
    order_id: str = Field(description="Order ID to modify")
    modification_type: str = Field(description="Type: add_item, remove_item, change_quantity, special_request")
    item_details: Optional[Dict[str, Any]] = Field(default=None, description="Item details for modification")
    reason: Optional[str] = Field(default=None, description="Reason for modification")

@tool(
    name_or_callable="modify_order_api",
    description=get_tool_description('modify_order_api'),
    args_schema=OrderModification,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def modify_order_api(order_id: str, modification_type: str, item_details: Optional[Dict[str, Any]] = None, 
                          reason: Optional[str] = None, config: RunnableConfig = None) -> str:
    """Modify order in kitchen system"""
    
    # Check if modification is possible (mock logic)
    modification_possible = True
    additional_time = 0
    
    if modification_type == "add_item":
        additional_time = 10
        message = f"Item added to order {order_id}"
    elif modification_type == "remove_item":
        message = f"Item removed from order {order_id}"
    elif modification_type == "change_quantity":
        additional_time = 5
        message = f"Quantity updated for order {order_id}"
    elif modification_type == "special_request":
        additional_time = 8
        message = f"Special request added to order {order_id}"
    else:
        modification_possible = False
        message = "Invalid modification type"
    
    response = {
        "order_id": order_id,
        "modification_successful": modification_possible,
        "message": message,
        "additional_prep_time": f"{additional_time} minutes" if additional_time > 0 else "No additional time",
        "updated_eta": "Updated timing will be provided shortly"
    }
    
    if reason:
        response["modification_reason"] = reason
    
    return str(response)


# =============================================================================
# BILLING AND PAYMENT TOOLS
# =============================================================================

class BillGeneration(BaseModel):
    order_id: str = Field(description="Order ID for billing")
    table_number: str = Field(description="Table number")
    discount_code: Optional[str] = Field(default=None, description="Discount code if applicable")
    split_bill: Optional[int] = Field(default=1, description="Number of ways to split the bill")

@tool(
    name_or_callable="generate_bill_api",
    description=get_tool_description('generate_bill_api'),
    args_schema=BillGeneration,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def generate_bill_api(order_id: str, table_number: str, discount_code: Optional[str] = None, 
                           split_bill: int = 1, config: RunnableConfig = None) -> str:
    """Generate detailed bill for order"""
    
    bill_id = f"BILL_{uuid.uuid4().hex[:8].upper()}"
    
    # Mock order items for billing
    order_items = [
        {"name": "Grilled Salmon", "price": 24.99, "quantity": 1},
        {"name": "House Wine", "price": 8.99, "quantity": 2},
        {"name": "Caesar Salad", "price": 12.99, "quantity": 1}
    ]
    
    subtotal = sum(item["price"] * item["quantity"] for item in order_items)
    
    # Apply discount
    discount_amount = 0
    if discount_code:
        if discount_code.upper() == "HAPPY10":
            discount_amount = subtotal * 0.10
        elif discount_code.upper() == "STUDENT15":
            discount_amount = subtotal * 0.15
    
    subtotal_after_discount = subtotal - discount_amount
    tax_rate = 0.08  # 8% tax
    tax_amount = subtotal_after_discount * tax_rate
    service_charge = subtotal_after_discount * 0.15  # 15% service
    total = subtotal_after_discount + tax_amount + service_charge
    
    bill_data = {
        "bill_id": bill_id,
        "order_id": order_id,
        "table_number": table_number,
        "items": order_items,
        "subtotal": round(subtotal, 2),
        "discount": round(discount_amount, 2) if discount_amount > 0 else 0,
        "subtotal_after_discount": round(subtotal_after_discount, 2),
        "tax": round(tax_amount, 2),
        "service_charge": round(service_charge, 2),
        "total": round(total, 2),
        "payment_methods": ["cash", "card", "digital_wallet", "mobile_pay"],
        "split_bill_count": split_bill,
        "amount_per_person": round(total / split_bill, 2) if split_bill > 1 else round(total, 2)
    }
    
    if discount_code and discount_amount > 0:
        bill_data["discount_code_applied"] = discount_code
    
    return str(bill_data)


class PaymentProcessing(BaseModel):
    bill_id: str = Field(description="Bill ID for payment")
    payment_method: str = Field(description="Payment method: cash, card, digital_wallet, mobile_pay")
    amount: float = Field(description="Amount to process")
    tip_amount: Optional[float] = Field(default=None, description="Tip amount if applicable")

@tool(
    name_or_callable="process_payment_api",
    description=get_tool_description('process_payment_api'),
    args_schema=PaymentProcessing,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def process_payment_api(bill_id: str, payment_method: str, amount: float, 
                             tip_amount: Optional[float] = None, config: RunnableConfig = None) -> str:
    """Process payment through POS system"""
    
    transaction_id = f"TXN_{uuid.uuid4().hex[:8].upper()}"
    
    payment_response = {
        "transaction_id": transaction_id,
        "bill_id": bill_id,
        "payment_method": payment_method,
        "amount_paid": amount,
        "tip_amount": tip_amount or 0,
        "total_charged": amount + (tip_amount or 0),
        "status": "completed",
        "timestamp": "2025-08-03T14:30:00Z",
        "receipt_number": f"RCP_{transaction_id[-6:]}",
        "payment_confirmation": "Payment processed successfully"
    }
    
    if payment_method == "card":
        payment_response["card_details"] = {
            "last_four": "****1234",
            "card_type": "Visa",
            "approval_code": f"APP{uuid.uuid4().hex[:6].upper()}"
        }
    elif payment_method == "digital_wallet":
        payment_response["digital_wallet"] = {
            "provider": "Apple Pay",
            "transaction_reference": f"AP_{uuid.uuid4().hex[:8]}"
        }
    
    return str(payment_response)


# =============================================================================
# TABLE AND RESERVATION TOOLS
# =============================================================================

class TableStatusQuery(BaseModel):
    table_number: Optional[str] = Field(default=None, description="Specific table number to check")
    area: Optional[str] = Field(default=None, description="Restaurant area: main_dining, patio, bar")

@tool(
    name_or_callable="check_table_status_api",
    description=get_tool_description('check_table_status_api'),
    args_schema=TableStatusQuery,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def check_table_status_api(table_number: Optional[str] = None, area: Optional[str] = None, 
                                config: RunnableConfig = None) -> str:
    """Get current status of restaurant tables"""
    
    # Mock table data
    table_statuses = {
        "1": {"status": "occupied", "party_size": 4, "seated_time": "1:30 PM", "estimated_turnover": "2:45 PM"},
        "2": {"status": "available", "last_cleaned": "2:00 PM", "capacity": 2},
        "3": {"status": "reserved", "reservation_time": "3:00 PM", "party_name": "Johnson", "party_size": 6},
        "4": {"status": "needs_cleaning", "last_occupied": "1:45 PM", "capacity": 4},
        "5": {"status": "occupied", "party_size": 2, "seated_time": "2:15 PM", "estimated_turnover": "3:30 PM"}
    }
    
    if table_number:
        if table_number in table_statuses:
            table_info = table_statuses[table_number]
            table_info["table_number"] = table_number
            return str(table_info)
        else:
            return f"Table {table_number} not found"
    else:
        # Return all tables status
        all_tables = []
        for table_num, status in table_statuses.items():
            table_info = status.copy()
            table_info["table_number"] = table_num
            all_tables.append(table_info)
        
        return str({"tables": all_tables, "total_tables": len(all_tables)})


class ReservationQuery(BaseModel):
    date: str = Field(description="Date for reservation (YYYY-MM-DD)")
    time: Optional[str] = Field(default=None, description="Specific time (HH:MM)")
    party_size: Optional[int] = Field(default=None, description="Number of people")

@tool(
    name_or_callable="check_reservation_availability_api",
    description=get_tool_description('check_reservation_availability_api'),
    args_schema=ReservationQuery,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def check_reservation_availability_api(date: str, time: Optional[str] = None, party_size: Optional[int] = None, 
                                           config: RunnableConfig = None) -> str:
    """Check reservation availability for specified date/time"""
    
    # Mock availability data
    available_slots = [
        {"time": "5:00 PM", "available_tables": 3, "max_party_size": 6},
        {"time": "5:30 PM", "available_tables": 2, "max_party_size": 4},
        {"time": "6:00 PM", "available_tables": 4, "max_party_size": 8},
        {"time": "6:30 PM", "available_tables": 1, "max_party_size": 2},
        {"time": "7:00 PM", "available_tables": 2, "max_party_size": 6},
        {"time": "7:30 PM", "available_tables": 3, "max_party_size": 4},
        {"time": "8:00 PM", "available_tables": 2, "max_party_size": 6}
    ]
    
    # Filter by party size if specified
    if party_size:
        available_slots = [slot for slot in available_slots if slot["max_party_size"] >= party_size]
    
    # Filter by specific time if requested
    if time:
        available_slots = [slot for slot in available_slots if slot["time"] == time]
    
    reservation_info = {
        "date": date,
        "requested_time": time,
        "requested_party_size": party_size,
        "available_slots": available_slots,
        "total_available": len(available_slots)
    }
    
    return str(reservation_info)


# =============================================================================
# CUSTOMER SERVICE TOOLS
# =============================================================================

class CustomerFeedback(BaseModel):
    table_number: str = Field(description="Table number")
    feedback_type: str = Field(description="Type: complaint, compliment, suggestion, allergy_alert")
    message: str = Field(description="Customer feedback message")
    urgency: str = Field(default="normal", description="Urgency level: low, normal, high, urgent")

@tool(
    name_or_callable="log_customer_feedback_api",
    description=get_tool_description('log_customer_feedback_api'),
    args_schema=CustomerFeedback,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def log_customer_feedback_api(table_number: str, feedback_type: str, message: str, urgency: str = "normal", 
                                   config: RunnableConfig = None) -> str:
    """Record customer feedback in management system"""
    
    feedback_id = f"FB_{uuid.uuid4().hex[:8].upper()}"
    
    feedback_record = {
        "feedback_id": feedback_id,
        "table_number": table_number,
        "feedback_type": feedback_type,
        "message": message,
        "urgency": urgency,
        "timestamp": "2025-08-03T14:30:00Z",
        "status": "logged",
        "assigned_to": "Floor Manager" if urgency in ["high", "urgent"] else "Service Team",
        "follow_up_required": urgency in ["high", "urgent"]
    }
    
    if feedback_type == "complaint":
        feedback_record["escalation_level"] = 1
        feedback_record["response_deadline"] = "Within 15 minutes"
    elif feedback_type == "allergy_alert":
        feedback_record["alert_kitchen"] = True
        feedback_record["urgency"] = "urgent"
    
    return str(feedback_record)


class SpecialRequest(BaseModel):
    table_number: str = Field(description="Table number making request")
    request_type: str = Field(description="Type: dietary, seating, service, celebration")
    details: str = Field(description="Specific request details")
    
@tool(
    name_or_callable="handle_special_request_api",
    description=get_tool_description('handle_special_request_api'),
    args_schema=SpecialRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def handle_special_request_api(table_number: str, request_type: str, details: str, 
                                    config: RunnableConfig = None) -> str:
    """Handle special customer requests"""
    
    request_id = f"REQ_{uuid.uuid4().hex[:8].upper()}"
    
    request_record = {
        "request_id": request_id,
        "table_number": table_number,
        "request_type": request_type,
        "details": details,
        "status": "processing",
        "feasible": True,
        "estimated_fulfillment": "5-10 minutes",
        "assigned_staff": "Available waiter"
    }
    
    # Handle different request types
    if request_type == "dietary":
        request_record["kitchen_notified"] = True
        request_record["chef_approval"] = "Required"
    elif request_type == "celebration":
        request_record["items_needed"] = ["celebration_music", "special_dessert"]
        request_record["estimated_fulfillment"] = "10-15 minutes"
    elif request_type == "seating":
        request_record["host_notified"] = True
        request_record["table_change_possible"] = True
    
    return str(request_record)
