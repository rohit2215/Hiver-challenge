from typing import List
from pydantic import BaseModel, Field
from config import client, MODEL_NAME

class MetricBreakdown(BaseModel):
    factual_completeness: float = Field(
        ...,
        description="Factual completeness score between 0.0 and 1.0. Checks if all ground truth resolution points were explicitly met in the generated reply. 1.0 means all points are completely addressed, 0.0 means none are addressed."
    )
    tone_and_safety: float = Field(
        ...,
        description="Tone and safety score between 0.0 and 1.0. Validates professionalism, empathy, tier-appropriate treatment, and absolute freedom from hallucinated or toxic statements. 1.0 means perfect, safe tone."
    )
    actionability: float = Field(
        ...,
        description="Actionability score between 0.0 and 1.0. Ensures that next steps for both Hiver and the client are clearly stated and actionable. 1.0 means extremely clear next steps."
    )
    reasoning: str = Field(
        ...,
        description="A detailed text auditing explanation justifying the scores, detailing which specific ground truth points were met/unmet, and grading the tone and actionability."
    )

def evaluate_response_quality(incoming_email: str, generated_reply: str, ground_truth_points: List[str]) -> MetricBreakdown:
    """
    Evaluates the quality of a generated reply against an incoming email and its ground truth points.
    Uses LLM-as-a-Judge with strict Pydantic structured output.
    """
    formatted_gt_points = "\n".join(f"- {pt}" for pt in ground_truth_points)

    system_prompt = """You are an expert Quality Assurance Judge auditing customer support emails at Hiver. 
Your goal is to evaluate a drafted support reply against the customer's incoming email and a set of explicit, mandatory ground truth resolution points.

Evaluate the draft response strictly according to the following dimensions:

1. Factual Completeness (0.0 to 1.0):
   Check if every single ground truth resolution point is covered in the generated reply. 
   - A score of 1.0 means all points are fully and accurately addressed.
   - Calculate this mathematically: (number of fully met points + 0.5 * number of partially met points) / total number of points.

2. Tone and Safety (0.0 to 1.0):
   Evaluate if the response is professional, empathetic, and safe (i.e. free of toxic language, anger, sarcasm, or hallucinations/promises not supported by the ground truth).
   - The response should validate the customer's frustration appropriately.
   - It should NOT hallucinate names, dates, or terms that are unsupported.

3. Actionability (0.0 to 1.0):
   Evaluate if the reply provides clear, concrete next steps for both the customer and the support team. 
   - If the next steps are vague or missing, assign a lower score.

Provide a detailed, step-by-step audit in the 'reasoning' field and assign float scores for each dimension. Ensure you are objective, strict, and consistent.
"""

    user_prompt = f"""[Incoming Email]
{incoming_email}

[Ground Truth Resolution Points]
{formatted_gt_points}

[Generated Suggested Reply]
{generated_reply}

Please perform your audit and return the structured metric breakdown:"""

    # Call OpenAI Structured Outputs API with temperature 0.0 for maximum consistency
    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=MetricBreakdown,
        temperature=0.0  # Force deterministic grading
    )

    parsed_result = completion.choices[0].message.parsed
    if not parsed_result:
        raise RuntimeError("Failed to parse evaluation metrics from the LLM judge.")

    return parsed_result
