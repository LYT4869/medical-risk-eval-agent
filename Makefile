.PHONY: prepare-demo demo demo-real demo-observability demo-flow verify verify-full \
	routing-unit routing-calibrate routing-evaluate routing-load-smoke

ROUTING_MODEL := intfloat/multilingual-e5-small
ROUTING_REVISION := 614241f622f53c4eeff9890bdc4f31cfecc418b3
ROUTING_ROOT := PythonServices/TreeSemAgent
ROUTING_CASES := $(ROUTING_ROOT)/evaluation/routing_cases.json
ROUTING_TASKS := $(ROUTING_ROOT)/config/tasks.yaml
ROUTING_THRESHOLDS := $(ROUTING_ROOT)/config/routing_thresholds.json

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

routing-unit:
	PYTHONPATH=$(ROUTING_ROOT) python3 -m unittest discover -s $(ROUTING_ROOT)/tests -p 'test_routing*.py' -v

routing-calibrate:
	PYTHONPATH=$(ROUTING_ROOT) python3 $(ROUTING_ROOT)/evaluation/calibrate_routing.py --cases $(ROUTING_CASES) --tasks $(ROUTING_TASKS) --model $(ROUTING_MODEL) --revision $(ROUTING_REVISION) --output $(ROUTING_THRESHOLDS)

routing-evaluate:
	PYTHONPATH=$(ROUTING_ROOT) python3 $(ROUTING_ROOT)/evaluation/run_routing_evaluation.py --router hybrid --split held_out --cases $(ROUTING_CASES) --tasks $(ROUTING_TASKS) --thresholds $(ROUTING_THRESHOLDS) --model $(ROUTING_MODEL) --revision $(ROUTING_REVISION) --output artifacts/evaluation/routing/hybrid-held-out.json

routing-load-smoke:
	@for round in $$(seq 1 20); do \
		TREESEM_TRACE_STDOUT=false PYTHONPATH=$(ROUTING_ROOT) python3 -m unittest tests.test_routing_executor >/dev/null || exit 1; \
	done
	@echo "routing executor load smoke passed 20 rounds"
