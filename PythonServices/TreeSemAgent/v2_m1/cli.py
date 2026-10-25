"""Synthetic CLI: each invocation is a new process with the same SQLite files."""
import argparse
import json

from .service import DemoWorkflowService


def main():
    p = argparse.ArgumentParser(description="Synthetic-only TreeSem V2 M1 HITL")
    p.add_argument("--data-dir", required=True)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("start")
    check = sub.add_parser("status")
    check.add_argument("run_id")
    for name, options in (("confirm", ["confirm", "cancel", "revise"]),
                          ("review", ["acknowledge", "cancel"])):
        cmd = sub.add_parser(name)
        cmd.add_argument("run_id")
        cmd.add_argument("--decision", choices=options, required=True)
        cmd.add_argument("--revision", type=int, required=True)
    args = p.parse_args()
    service = DemoWorkflowService(args.data_dir)
    if args.command == "start":
        result = service.start()
    elif args.command == "status":
        result = service.status(args.run_id)
    else:
        result = service.resume(args.run_id, "contract" if args.command == "confirm" else "review",
                                args.decision, args.revision)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
