# ChatreeDeka POC

ChatreeDeka is a CLI-based Thai civil courtroom simulator built as a proof-of-concept using LangGraph. It simulates a courtroom with roles for Judge, Prosecutor, and Defender, enforcing procedural rules through graph edges and RAG rather than pure LLM prompting.

## Installation

```bash
pip install -r requirements.txt
```

Optional observability (LangSmith tracing):
```bash
pip install -e ".[observability]"
```

Set up `.env` from `.env.example` with your API keys.

## Quick Start

**Interactive simulation (default):**
```bash
python -m src.chatree_deka.cli simulate --case-file case-file/1.json
```

**Flags:**
- `--case-file` (required): Path to case JSON
- `--mode` [run|train|coached]: Execution mode (default: run)
- `--prosecutor` [manual|ai]: Prosecutor role (default: ai)
- `--defender` [manual|ai]: Defender role (default: ai)
- `--judge` [manual|ai]: Judge role (default: ai)
- `--output-file`: JSON output path (default: transcript_output.json)

**Examples:**

Interactive with manual operators:
```bash
python -m src.chatree_deka.cli simulate --case-file case-file/1.json \
  --prosecutor manual --defender manual --judge ai
```

Headless self-play training (50 episodes, configured in `config/marti_config.yaml`):
```bash
python -m src.chatree_deka.cli train --case-file case-file/1.json --episodes 100
```

Coached mode (human expert plays one role, model learns):
```bash
python -m src.chatree_deka.cli simulate --case-file case-file/1.json \
  --mode coached --prosecutor manual --defender ai
```

## Configuration

Edit `config/config.yaml` to set:
- **Models:** Ollama endpoints for prosecutor, defender, judge, evaluator
- **Inference:** Max tokens, temperature, retry limits
- **Monitoring:** LangSmith tracing (optional; toggled by `langsmith_enabled`)
- **Training:** Reward weights, MARTI config path, checkpoints

## MARL Training (MARTI)

ChatreeDeka integrates MARTI for multi-agent reinforcement learning self-play.

**Training mode:**
- Runs `N` headless episodes (configured in `config/marti_config.yaml`)
- Suppresses all manual interrupts (all roles forced to `ai`)
- Computes per-episode reward: `verdict_weight × verdict_outcome + compliance_weight × compliance_rate`
- Checkpoints policy weights to `training/checkpoints/`

**Reward function:**
- Verdict determined by judge agent using Thailand legal code analysis (via RAG)
- Prosecutor wins → reward = 1.0; otherwise 0.0
- Compliance bonus: `compliance_rate` (valid turns / total turns) for both roles
- Weights in `config/config.yaml` under `training.reward_weights`

**Coached mode:**
- Same as interactive (`run`) but episodes are fed to MARTI for policy updates
- Enables learning from human expert play

## Architecture

- **Orchestration:** LangGraph StateGraph with typed state machine
- **LLM Gateway:** LiteLLM → local Ollama (no external API calls required)
- **Validation:** Pydantic schemas + LlamaIndex-based legal citation grounding (TCCC)
- **RAG:** ChromaDB vector store chunked by Thai civil code section (`มาตรา`)
- **Verdict Determination:** Judge agent analyzes case facts, legal code, and transcript to determine winner and confidence
- **Observability:** Optional LangSmith tracing (config-toggled)

## Resources & Assets
Check out our [GitHub Wiki](https://github.com/ChatreeDeka/court-simulation/wiki) for addtional resources & assets (e.g., Dataset, Extraction Pipeline, Report)
