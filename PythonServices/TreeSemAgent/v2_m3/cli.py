"""M3 CLI: fake deterministic by default; opt-in local OpenAI-compatible model.

Do not run this with real patient data or against a production EHR.
"""
from __future__ import annotations

import argparse
import json
import os

from v2_m2.gateway_adapter import GatewayTransport, GatewayArtifactRepository
from .agent import FakeInvestigationAgent, OpenAICompatibleInvestigationAgent
from .service import M3LocalWorkflow, M3GatewayWorkflow


def main():
    parser = argparse.ArgumentParser(description="TreeSem V2 M3 synthetic Agenda")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mode", choices=("local", "gateway"), default="local")
    parser.add_argument("--agent", choices=("fake", "local_llm"), default="fake")
    parser.add_argument("run_id")
    parser.add_argument("--contract-revision", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16)
    args = parser.parse_args()

    if args.agent == "local_llm":
        url = os.environ["TREESEM_M3_LLM_ENDPOINT"]
        model = os.environ["TREESEM_M3_LLM_MODEL"]
        decision_model = OpenAICompatibleInvestigationAgent(
            url, model, bearer_token=os.environ.get("TREESEM_M3_LLM_TOKEN"),
        )
    else:
        decision_model = FakeInvestigationAgent()
    if args.mode == "gateway":
        transport = GatewayTransport(os.environ["TREESEM_M3_GATEWAY_URL"])
        repository = GatewayArtifactRepository(
            transport, os.environ["TREESEM_M3_CAPABILITY_TOKEN"]
        )
        runner = M3GatewayWorkflow(args.data_dir, repository, agent=decision_model)
    else:
        runner = M3LocalWorkflow(args.data_dir, agent=decision_model)
    outcome = runner.execute(
        args.run_id, args.contract_revision, max_steps=args.max_steps,
    )
    print(json.dumps(outcome, ensure_ascii=False))


if __name__ == "__main__":
    main()
