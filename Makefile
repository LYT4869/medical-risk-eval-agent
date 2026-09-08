.PHONY: prepare-demo demo demo-real demo-observability demo-flow verify verify-full \
	routing-unit routing-calibrate routing-evaluate routing-load-smoke \
	routing-export-fp32 routing-export-int8 routing-parity routing-benchmark

ROUTING_MODEL := intfloat/multilingual-e5-small
ROUTING_REVISION := 614241f622f53c4eeff9890bdc4f31cfecc418b3
ROUTING_ROOT := PythonServices/TreeSemAgent
ROUTING_CASES := $(ROUTING_ROOT)/evaluation/routing_cases.json
ROUTING_TASKS := $(ROUTING_ROOT)/config/tasks.yaml
ROUTING_THRESHOLDS := $(ROUTING_ROOT)/config/routing_thresholds.json
ROUTING_ARTIFACT_DIR ?= artifacts/agent-routing/disabled
ROUTING_BACKEND ?= onnx_int8
ROUTING_HF_CACHE ?= $(HOME)/.cache/huggingface
ROUTING_EXPORT_IMAGE ?= treesem-routing-export:local
ROUTING_REPORT_DIR := artifacts/evaluation/routing

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

routing-export-fp32:
	TREESEM_ROUTING_EXPORT_BACKEND=onnx_fp32 \
	TREESEM_ROUTING_EXPORT_IMAGE=$(ROUTING_EXPORT_IMAGE) \
	TREESEM_HF_CACHE_DIR=$(ROUTING_HF_CACHE) \
	./scripts/prepare-routing-model.sh

routing-export-int8:
	TREESEM_ROUTING_EXPORT_BACKEND=onnx_int8 \
	TREESEM_ROUTING_EXPORT_IMAGE=$(ROUTING_EXPORT_IMAGE) \
	TREESEM_HF_CACHE_DIR=$(ROUTING_HF_CACHE) \
	./scripts/prepare-routing-model.sh

routing-parity:
	mkdir -p $(ROUTING_REPORT_DIR)
	docker build --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
		--build-arg NO_PROXY -f deploy/docker/routing-export.Dockerfile \
		-t $(ROUTING_EXPORT_IMAGE) .
	docker run --rm --user "$$(id -u):$$(id -g)" --entrypoint python \
		-e HF_HOME=/models/huggingface -e HF_HUB_OFFLINE=1 \
		-e TRANSFORMERS_OFFLINE=1 \
		-v $(abspath $(ROUTING_HF_CACHE)):/models/huggingface:ro \
		-v $(abspath $(ROUTING_ARTIFACT_DIR)):/routing/artifact:ro \
		-v $(abspath $(ROUTING_REPORT_DIR)):/reports \
		$(ROUTING_EXPORT_IMAGE) -m evaluation.compare_routing_backends \
		--cases /app/evaluation/routing_cases.json \
		--tasks /app/config/tasks.yaml \
		--thresholds /app/config/routing_thresholds.json \
		--model $(ROUTING_MODEL) --revision $(ROUTING_REVISION) \
		--artifact-dir /routing/artifact --embedding-backend $(ROUTING_BACKEND) \
		--output /reports/$(ROUTING_BACKEND)-parity.json

routing-benchmark:
	PYTHONPATH=$(ROUTING_ROOT) python3 -m evaluation.benchmark_routing_runtime \
		--artifact-dir $(ROUTING_ARTIFACT_DIR) --tasks $(ROUTING_TASKS) \
		--thresholds $(ROUTING_THRESHOLDS) --backend $(ROUTING_BACKEND) \
		--warmup 100 --iterations 1000 \
		--output $(ROUTING_REPORT_DIR)/$(ROUTING_BACKEND)-runtime-benchmark.json
