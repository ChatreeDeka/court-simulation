from __future__ import annotations
import typer
import json
import time
from rich import print
from langgraph.checkpoint.memory import MemorySaver
from chatree_deka.graph.builder import build_graph
from chatree_deka.state import TrialState
from chatree_deka.io.transcript_writer import write_transcript_to_json

app = typer.Typer(help="ChatreeDeka POC CLI Courtroom Simulator")

@app.callback()
def main():
    pass

@app.command()
def simulate(
    case_file: str = typer.Option(..., help="Path to case facts JSON"),
    prosecutor: str = typer.Option("ai", help="[manual|ai]"),
    defender: str = typer.Option("ai", help="[manual|ai]"),
    judge: str = typer.Option("ai", help="[manual|ai]"),
    output_file: str = typer.Option("transcript_output.json", help="Path to save the structured JSON transcript")
):
    print("[bold green]Starting ChatreeDeka Simulation...[/bold green]")
    try:
        with open(case_file, "r", encoding="utf-8") as f:
            case_data = json.load(f)
            if isinstance(case_data, list) and len(case_data) > 0:
                case_data = case_data[0]
            
            if "case_facts" not in case_data:
                facts = (
                    f"โจทก์: {case_data.get('plaintiff', 'ไม่ระบุ')}\n"
                    f"จำเลย: {case_data.get('defendant', 'ไม่ระบุ')}\n"
                    f"คำฟ้อง/คำร้อง: {case_data.get('plaintiff_claim', 'ไม่ระบุ')}\n"
                    f"ข้อเท็จจริงที่ยุติแล้ว: {case_data.get('undisputed_facts', 'ไม่ระบุ')}"
                )
                case_data["case_facts"] = facts
    except Exception as e:
        print(f"[bold red]Failed to load case file: {e}[/bold red]")
        # Placeholder mock load if file doesn't exist
        case_data = {"case_id": "test_1", "case_facts": "โจทย์ฟ้องเรียกค่าเสียหายฐานละเมิด"}
        print("[yellow]Using placeholder case data...[/yellow]")

    # Build Graph with Checkpointer
    checkpointer = MemorySaver()
    graph_with_mem = build_graph(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": "session_1"}}

    state: TrialState = {
        "case_id": case_data.get("case_id", "001"),
        "phase": "opening_prosecution",
        "judge_mode": judge,
        "prosecutor_mode": prosecutor,
        "defender_mode": defender,
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
        "current_speaker": "prosecutor" # First up
    }
    
    # Push initial
    for event in graph_with_mem.stream(state, thread_config):
        pass

    while True:
        current_state = graph_with_mem.get_state(thread_config)
        next_nodes = current_state.next
        if not next_nodes:
            break
            
        cur_node = next_nodes[0]
        role = cur_node.replace("_node", "")
        # role can be phase_router_after_val, but it won't interrupt
        mode = current_state.values.get(f"{role}_mode", "ai")
        phase = current_state.values.get("phase", "unknown")
        
        # Check if we should block for user input
        if mode == "manual" and role in ["prosecutor", "defender", "judge"]:
            print(f"\n[bold blue][PHASE: {phase} — {role.capitalize()}'s Turn][/bold blue]")
            
            # Print latest transcript logic here if needed
            operator_input = typer.prompt("Enter your statement (or /object, /evidence <text>, /end)")
            
            if operator_input.startswith("/object"):
                graph_with_mem.update_state(thread_config, {"objection_pending": True})
                print("[yellow]Objection logged! The judge will rule next turn.[/yellow]")
                # We do not append the command to statement. We give empty statement.
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

        # Run turn        
        for event in graph_with_mem.stream(None, thread_config):
            # Print intermediate LLM outputs if validating
            for k, v in event.items():
                if k in ["prosecutor_node", "defender_node", "judge_node", "evaluator_node"]:
                    stmt = v.get("pending_statement")
                    if stmt:
                        print(f"\n[bold magenta]{k.capitalize()}:[/bold magenta] {stmt}")
                elif k == "validation_node":
                    if v.get("validation_result") == "fail":
                        print(f"[bold red]Validation Failed:[/bold red] {v.get('validation_reason')}")
        
        time.sleep(5)

    print("\n[bold green]Simulation Ended. Final Transcript:[/bold green]")
    final_state = graph_with_mem.get_state(thread_config).values
    transcript_list = final_state.get("transcript", [])
    for turn in transcript_list:
        print(f"{turn['role'].capitalize()}: {turn['content']}")

    if final_state.get("evaluation_result"):
        print(f"\n[bold cyan]Evaluation Result:[/bold cyan] {final_state.get('evaluation_result')}")
    if final_state.get("evaluation_reason"):
        print(f"[cyan]{final_state.get('evaluation_reason')}[/cyan]")
        
    write_transcript_to_json(transcript_list, final_state.get("case_id", "unknown"), output_file)
    print(f"\n[cyan]Transcript saved to: {output_file}[/cyan]")

if __name__ == "__main__":
    app()
