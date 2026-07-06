import json
import os
from pydantic import BaseModel, Field
from config import client, MODEL_NAME

class SuggestedReply(BaseModel):
    reply_body: str = Field(
        description="The complete, professional, and empathetic email body addressing the customer's issues and providing a clear resolution/next steps."
    )

def generate_suggested_reply(new_email: str, category: str) -> str:
    """
    Generates a suggested email reply for a new customer support ticket.
    Uses In-Context Few-Shot RAG by retrieving historical tickets from dataset.json
    and injecting them as demonstrations in the prompt.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    
    # Load historical golden dataset
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            historical_data = json.load(f)
    except FileNotFoundError:
        historical_data = []
    
    # Select few-shot demonstrations matching the category
    examples = [t for t in historical_data if t.get("category") == category]
    if not examples:
        # Fallback to all examples if category doesn't match
        examples = historical_data

    # Format the examples into a string block
    few_shot_block = ""
    for idx, ex in enumerate(examples, 1):
        gt_points = "\n".join(f"- {pt}" for pt in ex.get("ground_truth_resolution_points", []))
        few_shot_block += f"""
=== HISTORICAL DEMONSTRATION {idx} ===
Category: {ex.get('category')}
Incoming Customer Email:
{ex.get('incoming_email')}

Required Resolution Checklist:
{gt_points}

Approved Historical Response:
{ex.get('historical_response')}
======================================
"""

    system_prompt = f"""You are an elite Customer Success Engineer at Hiver. Your task is to draft a professional, empathetic, and highly accurate suggested response to a new customer support email.

You must ground your response strictly in our approved historical responses and resolution principles. Do not invent arbitrary numbers, refund values, credit amounts, or resolution timelines that are not supported by the historical context provided.

Here is the relevant historical demonstration of an approved customer resolution:
{few_shot_block}

Tone Guidelines:
1. Show empathy: Acknowledge the issue, validate the customer's frustration, and understand the business impact.
2. Maintain high professionalism: Be clear, polite, and reassuring.
3. Be action-oriented: Clearly outline the steps we are taking and specify the next steps for the customer.
4. Ensure factual alignment: If the incoming request matches a demonstration category, apply the exact same logic (e.g., $100 credit for a Billing Discrepancy, 3-5 days processing; or temporary 24-hour limit bump to 800 RPM and recommend exponential backoff for an API rate limit issue). Do not use any generic placeholders.

Draft the suggested reply body and return it as structured output.
"""

    user_prompt = f"""Incoming Email Category: {category}
Incoming Email Content:
{new_email}

Please generate the suggested reply:"""

    # Call OpenAI Structured Outputs API
    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=SuggestedReply,
        temperature=0.1  # Low temperature for deterministic, factual output
    )

    # Extract structured response
    parsed_response = completion.choices[0].message.parsed
    if not parsed_response:
        raise RuntimeError("Failed to parse the structured output from the API.")
        
    return parsed_response.reply_body
