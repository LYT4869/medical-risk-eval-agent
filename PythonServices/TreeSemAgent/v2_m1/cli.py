"""Synthetic CLI: each invocation is a new process with the same SQLite files."""
import argparse
import json

from .service import DemoWorkflowService


def main():
    p = argparse.ArgumentParser(description="Synthetic-only TreeSem V2 M1 HITL")
    p.add_argument("--data-dir", required=True)
    sub = p.add_subparsers(dest="command", required=True)
    begin = sub.add_parser("start")
    begin.add_argument("--ask-clarification", action="store_true")
    check = sub.add_parser("status")
    check.add_argument("run_id")
    cmd = sub.add_parser("confirm")
    cmd.add_argument("run_id")
    cmd.add_argument("--decision", choices=["confirm", "revise", "clarify", "cancel"], required=True)
    cmd.add_argument("--revision", type=int, required=True)
    cmd.add_argument("--contract-revision", type=int, required=True)
    cmd.add_argument("--summary")
    cmd.add_argument("--overview")
    cmd.add_argument("--answer")
    args = p.parse_args()
    service = DemoWorkflowService(args.data_dir)
    if args.command == "start":
        result = service.start(ask_clarification=args.ask_clarification)
    elif args.command == "status":
        result = service.status(args.run_id)
    else:
        result = service.resume(args.run_id, args.decision, args.revision,
                                args.contract_revision, summary=args.summary,
                                overview=args.overview, answer=args.answer)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
