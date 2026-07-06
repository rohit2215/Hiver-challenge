# Suggested-Response System & Evaluation Pipeline

An end-to-end B2B customer support suggested-response generator and structured evaluation pipeline built for the Hiver challenge. The project uses Python 3.10+, OpenAI Structured Outputs via Pydantic, and In-Context Few-Shot RAG.

---

## 🏗️ Repository Layout

```text
├── .gitignore                   # Excludes credential and cached files
├── requirements.txt             # Python dependencies
├── config.py                    # Safely loads env files & exports OpenAI client
├── dataset.json                 # Historical golden dataset (2 B2B cases)
├── responder.py                 # Few-Shot RAG suggested reply generator
├── evaluator.py                 # LLM-as-a-Judge structured metric analyzer
├── main.py                      # Pipeline orchestrator
└── README.md                    # Production documentation
```

---

## ⚡ Setup & Execution

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Clone or copy these repository files to your local workspace, then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a file named `.env` in the root directory and add your OpenAI API key:
```env
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. Run the Pipeline Orchestrator
Execute the complete pipeline end-to-end:
```bash
python main.py
```

Upon execution, the script will:
- Process two simulated customer emails (Billing Discrepancy and API Integration Error).
- Generate answers grounded in historical customer support guidelines (`dataset.json`).
- Evaluate the responses via an LLM-as-a-Judge.
- Print detailed progress logs and final score summaries to the terminal.
- Export results to `evaluation_results.csv`.

---

## 🕵️ Hiring Team Evaluation Guide

This guide is designed for the hiring team to evaluate the correctness, modularity, and responsiveness of the system.

### 1. Verification of Non-Trivial Generation
To test how dynamically the system responds to custom support inquiries:
- Open [main.py](file:///Users/rahulsrivastava/Hiver-challenge/main.py) and modify the simulated emails in the `test_tickets` array (starting on line 8).
- Modify the `incoming_email` details or categories.
- Update the `ground_truth_points` to define the success parameters.
- Re-run `./venv/bin/python main.py` to check if the generator adapts its resolution steps and if the judge correctly scores the modifications.

### 2. Inspecting the Audit Trail (Factual Integrity)
Our pipeline saves a detailed audit trace in `evaluation_results.csv`:
- Open `evaluation_results.csv` in the repository root.
- Verify the `reasoning` column for each test case. You will find step-by-step rationales detailing exactly which ground truth points were matched, how the tone met the criteria, and whether the actionability checks succeeded.

### 3. Testing Missing Configurations
To test the error-handling mechanism:
- Temporarily rename your `.env` file to `.env.bak` and run the orchestrator:
  ```bash
  ./venv/bin/python main.py
  ```
- The application will raise a clean `ValueError: CRITICAL: OPENAI_API_KEY is missing from your .env file.` and exit immediately, proving the validation logic is secure.

---


## 🎓 Architecture Decisions & Trade-Offs

### 🧠 LLM-as-a-Judge vs. Lexical Metrics (BLEU / ROUGE)

In automated customer support systems, assessing reply quality using legacy lexical overlap metrics (like **BLEU** or **ROUGE**) is brittle and operationally dangerous. 

Here is why a semantic **LLM-as-a-Judge** architecture is superior:

| Dimension | Legacy Lexical Overlap (BLEU / ROUGE) | LLM-as-a-Judge (Semantic Boundaries) |
| :--- | :--- | :--- |
| **Semantic Understanding** | **Rigid token matching.** If a response uses synonyms (e.g., "refund" instead of "credit"), the score drops despite the resolution being correct. | **Conceptual evaluation.** Evaluates semantic intent rather than superficial string matches. |
| **Factual Validation** | **Incapable.** A sentence stating `"We will charge you $100"` has a very high ROUGE overlap with `"We will credit you $100"`, despite having the opposite business meaning. | **Checklist auditing.** Directly checks if specific resolution checkpoints (e.g., "$100 credit", "3-5 business days") are met. |
| **Tone and Style** | **Ignores tone.** Cannot detect if a generated response sounds rude, unprofessional, or lacks empathy. | **Tone Analysis.** Evaluates language safety, empathy levels, and tier-appropriate treatment. |
| **Actionability** | **Ignores logic.** Cannot determine if clear next steps are outlined for the user. | **Next-Step Audit.** Confirms if actionable paths are provided to the user. |
| **Explainability** | **Single numerical value.** Provides no rationale or context on why a response failed. | **Detailed Reasoning.** Returns structured explanation strings outlining why scores were given. |

---

### 🆚 In-Context Few-Shot RAG vs. Fine-Tuning

When designing a suggested-response system, selecting the right model alignment technique is crucial. Below is the comparative analysis of our chosen approach (**In-Context Few-Shot RAG**) versus **Fine-Tuning**:

| Feature / Metric | In-Context Few-Shot RAG (Our Choice) | Fine-Tuning |
| :--- | :--- | :--- |
| **Upfront Training Cost** | **Zero.** Uses existing base model API without extra compute overhead. | **High.** Requires computing resource leases, dataset preprocessing, and hyperparameter tuning. |
| **Deployment Agility** | **Instant.** Modifying corporate policy or adding historical examples is as simple as updating `dataset.json` or prompt instructions. | **Slow.** Requires preparing new datasets and redeploying a newly trained model checkpoint. |
| **Context Window Consumption** | **High.** Demonstrations consume system prompts tokens, increasing per-request billing and reducing input room. | **Zero.** The style and knowledge are baked into weights; prompts remain short and efficient. |
| **Out-of-Distribution Safety** | **High.** Employs clear constraints. The model refers strictly to injected context and is less prone to hallucinations. | **Moderate/Low.** Risk of "catastrophic forgetting" or hallucinating old patterns when queried on new topics. |
| **Auditability** | **Excellent.** You can see exactly which demonstration examples were injected into the prompt context. | **Poor.** Model updates behave like a black box; hard to isolate why specific weights fired a certain phrase. |

---

## 🛠️ Code Breakdown

### 1. `config.py`
Ensures secure execution by verifying environment keys. If `OPENAI_API_KEY` is missing, it crashes gracefully before starting the pipeline. It instantiates the global `client` using the OpenAI Python package and defines `MODEL_NAME = "gpt-4o-mini"`.

### 2. `responder.py`
Reads `dataset.json` and filters historical interactions by category. It injects these matches into a specialized prompt to instruct the model to follow specific business parameters (e.g., credit processing windows or temporary API limits). Using `temperature=0.1` ensures deterministic outputs.

### 3. `evaluator.py`
Leverages the Structured Outputs API by parsing responses directly into a structured Pydantic model (`MetricBreakdown`). Using a temperature of `0.0` ensures the judge remains objective and consistent across scoring evaluations.

### 4. `main.py`
The workflow orchestrator that processes test support tickets, calculates the final score using a weighted sum (`50% Factual + 25% Tone + 25% Actionability`), and prints a tabular summary along with an aggregate performance grid.

---

## 🤖 AI Tools & Technologies Used

This repository was designed, built, and tested utilizing the following AI tools and developer platforms:

- **OpenAI GPT-4o-mini:** Used as the generation engine for crafting context-grounded suggested replies and as the LLM Judge for evaluating response quality.
- **OpenAI Structured Outputs API:** Utilized via `client.beta.chat.completions.parse` to enforce strict Pydantic structures for deterministic scoring and structured reasoning auditing.
- **Antigravity AI Agent & Antigravity IDE:** Utilized for end-to-end codebase orchestration, context-aware package isolation (managing macOS PEP 668 constraints), test suite creation, and automated version control delivery.


