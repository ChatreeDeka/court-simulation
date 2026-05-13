from __future__ import annotations
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import typer
from rich import print
from langgraph.checkpoint.memory import MemorySaver
from chatree_deka.config import load_config
from chatree_deka.graph.builder import build_graph
from chatree_deka.state import TrialState
from chatree_deka.io.transcript_writer import write_transcript_to_json
from training.train import run_training

app = typer.Typer(help="ChatreeDeka POC CLI Courtroom Simulator")

@app.callback()
def main():
    load_dotenv()
    pass


def _format_case_facts(case_data: dict) -> str:
    parts = [
        f"โจทก์: {case_data.get('plaintiff', 'ไม่ระบุ')}",
        f"จำเลย: {case_data.get('defendant', 'ไม่ระบุ')}",
        f"คำฟ้อง/คำร้อง: {case_data.get('plaintiff_claim', 'ไม่ระบุ')}",
        f"ข้อเท็จจริงที่ยุติแล้ว: {case_data.get('undisputed_facts', 'ไม่ระบุ')}",
    ]

    if case_data.get("plaintiff_evidence"):
        parts.append(f"พยานหลักฐานฝ่ายโจทก์: {case_data.get('plaintiff_evidence')}")

    if case_data.get("defendant_evidence"):
        parts.append(f"พยานหลักฐานฝ่ายจำเลย: {case_data.get('defendant_evidence')}")

    if case_data.get("court_accepted_evidence"):
        parts.append(f"พยานหลักฐานที่ศาลรับฟัง: {case_data.get('court_accepted_evidence')}")

    if case_data.get("court_reasoning"):
        parts.append(f"เหตุผลศาล: {case_data.get('court_reasoning')}")

    if case_data.get("judgment"):
        parts.append(f"คำพิพากษา: {case_data.get('judgment')}")

    return "\n".join(str(part) for part in parts if part)


def configure_monitoring(cfg: dict) -> None:
    if not cfg.get("monitoring", {}).get("langsmith_enabled"):
        return

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.warning(
            "monitoring.langsmith_enabled is true but LANGCHAIN_API_KEY is not set. "
            "Tracing will be disabled for this session."
        )
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = cfg.get("monitoring", {}).get("langsmith_project", "chatree-deka-poc")


@app.command()
def simulate(
    case_file: str = typer.Option(..., help="Path to case facts JSON"),
    prosecutor: str = typer.Option("ai", help="[manual|ai]"),
    defender: str = typer.Option("ai", help="[manual|ai]"),
    judge: str = typer.Option("ai", help="[manual|ai]"),
    mode: str = typer.Option("run", help="[run|train|coached]"),
    output_file: str = typer.Option("transcript_output.json", help="Path to save the structured JSON transcript")
):
    print("[bold green]Starting ChatreeDeka Simulation...[/bold green]")
    cfg = load_config()
    configure_monitoring(cfg)

    mode = mode.lower()
    if mode not in ("run", "train", "coached"):
        raise typer.BadParameter("mode must be one of run, train, coached")

    if mode == "train":
        if any(role == "manual" for role in (prosecutor, defender, judge)):
            logger.warning(
                "Train mode forces all participants to AI. Manual role flags are ignored."
            )
        prosecutor = defender = judge = "ai"

    try:
        with open(case_file, "r", encoding="utf-8") as f:
            case_data = json.load(f)
            if isinstance(case_data, list) and len(case_data) > 0:
                case_data = case_data[0]

            if "case_facts" not in case_data:
                case_data["case_facts"] = _format_case_facts(case_data)
    except Exception as e:
        print(f"[bold red]Failed to load case file: {e}[/bold red]")
        case_data = {"case_id": "test_1", "case_facts": "โจทย์ฟ้องเรียกค่าเสียหายฐานละเมิด"}
        print("[yellow]Using placeholder case data...[/yellow]")

    checkpointer = MemorySaver()
    graph_with_mem = build_graph(checkpointer=checkpointer)
    thread_config = {"configurable": {"thread_id": "session_1"}}

    state: TrialState = {
        "case_id": case_data.get("case_id", "001"),
        "phase": "opening_prosecution",
        "judge_mode": judge,
        "prosecutor_mode": prosecutor,
        "defender_mode": defender,
        "mode": mode,
        "case_facts": case_data.get("case_facts", ""),
        "transcript": [],
        "pending_statement": None,
        "validation_result": None,
        "validation_reason": None,
        "retry_count": 0,
        "max_retries": 2,
        "evaluation_result": None,
        "evaluation_reason": None,
        "objection_pending": False,
        "current_speaker": "prosecutor"
    }

    for event in graph_with_mem.stream(state, thread_config):
        pass

    while True:
        current_state = graph_with_mem.get_state(thread_config)
        next_nodes = current_state.next
        if not next_nodes:
            break

        cur_node = next_nodes[0]
        role = cur_node.replace("_node", "")
        mode_value = current_state.values.get(f"{role}_mode", "ai")
        phase = current_state.values.get("phase", "unknown")

        if mode_value == "manual" and role in ["prosecutor", "defender", "judge"]:
            print(f"\n[bold blue][PHASE: {phase} — {role.capitalize()}'s Turn][/bold blue]")
            operator_input = typer.prompt("Enter your statement (or /object, /evidence <text>, /end)")

            if operator_input.startswith("/object"):
                graph_with_mem.update_state(thread_config, {"objection_pending": True})
                print("[yellow]Objection logged! The judge will rule next turn.[/yellow]")
                graph_with_mem.update_state(thread_config, {"pending_statement": None})
            elif operator_input.startswith("/evidence"):
                new_ev = operator_input.replace("/evidence", "").strip()
                old_facts = current_state.values.get("case_facts", "")
                graph_with_mem.update_state(thread_config, {"case_facts": old_facts + "\n" + new_ev})
                print("[cyan]Evidence updated![/cyan]")
                graph_with_mem.update_state(thread_config, {"pending_statement": None})
            elif operator_input == "/end":
                graph_with_mem.update_state(thread_config, {"phase": "verdict"})
                graph_with_mem.update_state(thread_config, {"pending_statement": None})
            else:
                graph_with_mem.update_state(thread_config, {"pending_statement": operator_input})

        for event in graph_with_mem.stream(None, thread_config):
            for k, v in event.items():
                if k in ["prosecutor_node", "defender_node", "judge_node", "evaluator_node"]:
                    stmt = v.get("pending_statement")
                    if stmt:
                        print(f"\n[bold magenta]{k.capitalize()}:[/bold magenta] {stmt}")
                elif k == "validation_node":
                    if v.get("validation_result") == "fail":
                        print(f"[bold red]Validation Failed:[/bold red] {v.get('validation_reason')}")

        time.sleep(5)

    final_state = graph_with_mem.get_state(thread_config).values
    transcript_list = final_state.get("transcript", [])

    if mode != "train":
        print("\n[bold green]Simulation Ended. Final Transcript:[/bold green]")
        for turn in transcript_list:
            print(f"{turn['role'].capitalize()}: {turn['content']}")

    if final_state.get("evaluation_result"):
        print(f"\n[bold cyan]Evaluation Result:[/bold cyan] {final_state.get('evaluation_result')}")
    if final_state.get("evaluation_reason"):
        print(f"[cyan]{final_state.get('evaluation_reason')}[/cyan]")

    summary = final_state.get("summary")
    if summary:
        print("\n[bold green]Trial Summary:[/bold green]")
        print(f"Mode: {summary.get('mode')}")
        print(f"Winner: {summary['verdict'].get('winner')} \nConfidence: {summary['verdict'].get('confidence')}")
        if summary.get("episode_reward") is not None:
            print(f"Episode Reward: {summary['episode_reward']}")

    write_transcript_to_json(
        transcript_list,
        final_state.get("case_id", "unknown"),
        output_file,
        summary=summary,
    )
    print(f"\n[cyan]Transcript saved to: {output_file}[/cyan]")


@app.command()
def train(
    case_file: str | None = typer.Option(
        None, help="Override training case file path from config"
    ),
    episodes: int | None = typer.Option(
        None, help="Override maximum episodes from config"
    ),
    checkpoint_path: str | None = typer.Option(
        None, help="Override checkpoint directory from config"
    ),
    checkpoint_every: int | None = typer.Option(
        None, help="Override checkpoint save frequency from config"
    ),
):
    print("[bold green]Starting ChatreeDeka Training...[/bold green]")
    cfg = load_config()
    configure_monitoring(cfg)
    run_training(
        case_file=case_file,
        episodes=episodes,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )


if __name__ == "__main__":
    app()
