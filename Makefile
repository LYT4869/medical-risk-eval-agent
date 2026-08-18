.PHONY: prepare-demo demo demo-real demo-observability demo-flow verify verify-full

prepare-demo:
	python3 scripts/prepare_demo.py

demo: prepare-demo
	docker compose up -d --build

demo-real: prepare-demo
	TREESEM_AGENT_LLM_MODE=real docker compose up -d --build

demo-observability: prepare-demo
	docker compose --profile observability up -d --build

demo-flow:
	python3 scripts/demo_flow.py

verify:
	ctest --test-dir build --output-on-failure
	PYTHONPATH=PythonServices/TreeSemAgent python3 PythonServices/TreeSemAgent/evaluation/run_evaluation.py --mode deterministic

verify-full:
	python3 scripts/verify_full.py --with-compose
