import pytest
from config import client, MODEL_NAME
from responder import generate_suggested_reply
from evaluator import evaluate_response_quality, MetricBreakdown

def test_config_initialization():
    """
    Verifies that configurations load correctly and the OpenAI client is instantiated.
    """
    assert MODEL_NAME == "gpt-4o-mini"
    assert client is not None
    assert client.api_key is not None

def test_responder_generates_response():
    """
    Tests that the suggested-response generator produces non-empty text when queried.
    """
    test_email = (
        "Subject: Charged $150 instead of $50 Growth tier flat rate. INV-99042.\n\n"
        "We only have 4,500 active contacts, which is below the 5k limit. Please refund."
    )
    category = "Billing Discrepancy"
    
    reply = generate_suggested_reply(test_email, category)
    
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0
    # Ensure key corporate parameters from historical resolution are captured
    assert "$100" in reply or "credit" in reply.lower()

def test_evaluator_grading_structure():
    """
    Tests that the evaluator outputs the exact Pydantic format with correct boundary constraints.
    """
    incoming_email = "Subject: Experiencing 429 rate limit errors on /v1/events"
    generated_reply = (
        "We have temporarily bumped your rate limit to 800 RPM for 24 hours. "
        "Please implement client-side exponential backoff to handle spikes."
    )
    ground_truth = [
        "Inform them that rate limit bumped to 800 RPM.",
        "Advise customer to implement client-side exponential backoff."
    ]
    
    metrics = evaluate_response_quality(
        incoming_email=incoming_email,
        generated_reply=generated_reply,
        ground_truth_points=ground_truth
    )
    
    assert isinstance(metrics, MetricBreakdown)
    # Check score ranges
    assert 0.0 <= metrics.factual_completeness <= 1.0
    assert 0.0 <= metrics.tone_and_safety <= 1.0
    assert 0.0 <= metrics.actionability <= 1.0
    # Check reasoning string exists
    assert isinstance(metrics.reasoning, str)
    assert len(metrics.reasoning.strip()) > 0
