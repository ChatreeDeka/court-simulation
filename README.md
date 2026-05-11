# ChatreeDeka POC

ChatreeDeka is a CLI-based Thai civil courtroom simulator built as a proof-of-concept using LangGraph. It simulates a courtroom with roles for Judge, Prosecutor, and Defender, enforcing procedural rules through graph edges rather than LLM prompting.

## Installation

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file (copy from `.env.example`):
   - `HF_TOKEN`: Your Hugging Face token for model access.

## How to Run

1. Ensure the config file `config/config.yaml` is set up with your desired models and settings.
2. Run the CLI:
   ```bash
   python -m src.chatree_deka.cli simulate --case-file ./case-file/1.json --output-file final_output.json --prosecutor manual --defender manual --judge manual
   ```
3. Follow the interactive prompts to simulate a court case. Use slash commands like `/help` for guidance.

## Configuration

### Using HuggingFace or Ollama Local Models

- Edit `config/config.yaml` to specify local models:
  - For HuggingFace: Set `model` to a local HF model path, e.g., `"microsoft/DialoGPT-medium"`. Ensure `load_strategy: sequential` to avoid VRAM issues.
  - For Ollama: Use Ollama-compatible model names via LiteLLM, e.g., `"ollama/llama2"`. Install Ollama locally and ensure it's running.
- All model calls go through `agents/base.py` using LiteLLM. No direct imports elsewhere.

### Injecting JSON Data (Historical Judgments and Legal Code)

- Place JSON files in `case-file/` directory, e.g., `1.json` for case data.
- For legal code and historical judgments, update the RAG system:
  - Add data to `rag/` components. Use `rag/chunker.py` to split Thai text by `มาตรา` boundaries.
  - The RAG retriever (`rag/retriever.py`) handles `semantic_search` for context and `section_lookup` for validation.
- Inject via CLI by loading case files or modify `cli.py` to accept custom inputs.

### Adjusting Agent Prompts

- Agent prompts are defined in `agents/*.py` files (e.g., `judge_agent.py`).
- Edit the system prompts in these files to customize behavior.
- Ensure changes align with the StateGraph architecture: prompts assemble context but don't handle routing.

## Usage

- Start a simulation with a case file.
- Agents interact in turns, with validation ensuring citations and grounding.
- Interrupt modes allow manual intervention when `{role}_mode` is set to "manual" in config.

## Development

- Read `docs/poc_spec.md` for technical details.
- Unit tests: `tests/unit/` (mock LLM calls).
- Integration tests: `tests/integration/` (mock LLMs, test graph).
- Acceptance test: `scripts/run_poc_test.py`.

For more details, see `docs/copilot-instructions.md`.