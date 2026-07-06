import os
import pandas as pd
from tabulate import tabulate
from responder import generate_suggested_reply
from evaluator import evaluate_response_quality

# Define the test tickets mimicking the categories in dataset.json
test_tickets = [
    {
        "id": "TEST-TICKET-001",
        "category": "Billing Discrepancy",
        "incoming_email": (
            "Subject: Dispute on Invoice INV-99042 - Double billing / overage fee?\n\n"
            "Hi Hiver Billing, our recent invoice INV-99042 says we owe $150. However, "
            "we are on the Growth tier, which is $50/month flat. It seems we were charged "
            "an overage fee of $100, but we only have 4,500 active contacts right now, "
            "which is below the 5k limit. Can you please look into this and correct it?\n\n"
            "Thanks,\nJohn Doe\nAdmin at Acme Corp"
        ),
        "ground_truth_points": [
            "Acknowledge that the customer was incorrectly billed $150 instead of the $50 Growth tier flat rate.",
            "Explain that the issue was due to an incorrect contact overage fee charge despite active contacts being below the limit.",
            "Confirm that a one-time $100 credit is being processed to resolve the discrepancy.",
            "State clearly that the credit processing will take approximately 3 to 5 business days to reflect on the account."
        ]
    },
    {
        "id": "TEST-TICKET-002",
        "category": "API Integration Error",
        "incoming_email": (
            "Subject: Event ingestion getting 429 Rate limits on /v1/events\n\n"
            "Hello Hiver Support, we are experiencing integration issues. Our event ingestion "
            "service is receiving 429 Too Many Requests errors when making POST requests to your `/v1/events`. "
            "This is breaking our live event ingestion pipeline. Our standard volume is normally "
            "fine, but we've seen a sudden spike in our traffic logs to 1200 RPM. Can you check "
            "our configuration and help us increase the limits or resolve this rate limit?\n\n"
            "Regards,\nAlice Smith\nTech Lead at BoltTech"
        ),
        "ground_truth_points": [
            "Explain that the HTTP 429 errors are occurring because requests spiked to 1,200 RPM, exceeding the standard rate limit of 600 RPM on the /v1/events endpoint.",
            "Inform the customer that their rate limit has been temporarily bumped to 800 RPM for the next 24 hours to mitigate the current issue.",
            "Advise the customer to implement client-side exponential backoff and retry mechanisms to gracefully handle rate limit exceptions.",
            "Suggest discussing a tier upgrade for a permanent limit increase if their baseline traffic remains above the standard limit."
        ]
    }
]

def main():
    print("=" * 60)
    print("Starting Hiver Email Response & Evaluation Pipeline Orchestrator")
    print("=" * 60)
    
    results = []

    for ticket in test_tickets:
        print(f"\n>>> Processing Ticket: {ticket['id']} ({ticket['category']})...")
        
        # 1. Generate suggested response
        print("    [1/3] Generating suggested reply (Few-Shot RAG)...")
        generated_reply = generate_suggested_reply(ticket["incoming_email"], ticket["category"])
        
        print("\n" + "-"*50)
        print(f"Generated Suggested Reply for {ticket['id']}:")
        print(generated_reply)
        print("-"*50 + "\n")
        
        # 2. Evaluate response quality
        print("    [2/3] Evaluating response quality (LLM-as-a-Judge)...")
        eval_breakdown = evaluate_response_quality(
            incoming_email=ticket["incoming_email"],
            generated_reply=generated_reply,
            ground_truth_points=ticket["ground_truth_points"]
        )
        
        # 3. Calculate composite overall score
        # Formula: 50% Factual + 25% Tone/Safety + 25% Actionability
        factual = eval_breakdown.factual_completeness
        tone = eval_breakdown.tone_and_safety
        actionable = eval_breakdown.actionability
        overall_score = (0.50 * factual) + (0.25 * tone) + (0.25 * actionable)
        
        print(f"    Evaluation Results: Factual={factual:.2f}, Tone={tone:.2f}, Actionability={actionable:.2f}")
        print(f"    Composite Overall Score: {overall_score:.2%}")
        print(f"    Judge Reasoning Summary: {eval_breakdown.reasoning.strip()}")
        
        # Store detailed results
        results.append({
            "ticket_id": ticket["id"],
            "category": ticket["category"],
            "incoming_email": ticket["incoming_email"],
            "generated_reply": generated_reply,
            "factual_completeness": factual,
            "tone_and_safety": tone,
            "actionability": actionable,
            "overall_score": overall_score,
            "reasoning": eval_breakdown.reasoning
        })

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)
    
    # 4. Save logs to CSV
    csv_filename = "evaluation_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n[Success] Processed {len(results)} tickets and saved details to '{csv_filename}'.")

    # 5. Output individual results as an ASCII grid via tabulate
    summary_table_data = []
    for r in results:
        summary_table_data.append([
            r["ticket_id"],
            r["category"],
            f"{r['factual_completeness']:.2f}",
            f"{r['tone_and_safety']:.2f}",
            f"{r['actionability']:.2f}",
            f"{r['overall_score']:.2%}"
        ])

    headers = ["Ticket ID", "Category", "Factual Score", "Tone & Safety", "Actionability", "Overall Score"]
    print("\n" + "=" * 80)
    print("                     PER-TICKET METRICS SUMMARY")
    print("=" * 80)
    print(tabulate(summary_table_data, headers=headers, tablefmt="grid"))
    print("=" * 80)

    # 6. Global aggregate averages
    avg_factual = df["factual_completeness"].mean()
    avg_tone = df["tone_and_safety"].mean()
    avg_actionability = df["actionability"].mean()
    avg_overall = df["overall_score"].mean()

    global_table_data = [
        ["Factual Completeness (50% Weight)", f"{avg_factual:.2f}"],
        ["Tone & Safety (25% Weight)", f"{avg_tone:.2f}"],
        ["Actionability (25% Weight)", f"{avg_actionability:.2f}"],
        ["Global Overall Performance", f"{avg_overall:.2%}"]
    ]

    print("\n" + "=" * 55)
    print("          GLOBAL AGGREGATE PERFORMANCE METRICS")
    print("=" * 55)
    print(tabulate(global_table_data, headers=["Metric Dimension", "Global Average Score"], tablefmt="grid"))
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
