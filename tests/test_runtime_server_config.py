import asyncio
import importlib
import os
import sys
from types import SimpleNamespace
import pytest

import genie_tts
from genie_tts import Internal, Server


def _load_real_model_manager_module():
    existing = sys.modules.get("genie_tts.ModelManager")
    if existing is not None and not getattr(existing, "__file__", None):
        sys.modules.pop("genie_tts.ModelManager", None)
    module = importlib.import_module("genie_tts.ModelManager")
    if not getattr(module, "__file__", None):
        sys.modules.pop("genie_tts.ModelManager", None)
        module = importlib.import_module("genie_tts.ModelManager")
    setattr(genie_tts, "ModelManager", module)
    setattr(genie_tts, "RuntimeConfig", module.RuntimeConfig)
    return module


ModelManagerModule = _load_real_model_manager_module()
ModelManager = ModelManagerModule.ModelManager
RuntimeConfig = ModelManagerModule.RuntimeConfig


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def refresh_real_model_manager_module():
    global ModelManagerModule, ModelManager, RuntimeConfig
    ModelManagerModule = _load_real_model_manager_module()
    ModelManager = ModelManagerModule.ModelManager
    RuntimeConfig = ModelManagerModule.RuntimeConfig


class TestRuntimeConfigAPI:
    def test_runtime_config_is_exported_from_package(self):
        assert "RuntimeConfig" in genie_tts.__all__
        runtime = RuntimeConfig(intra_op_num_threads=2)
        assert runtime.intra_op_num_threads == 2

    def test_model_manager_uses_runtime_config_providers(self, monkeypatch):
        manager = ModelManager()
        created = []

        monkeypatch.setattr(os.path, "exists", lambda path: path.endswith(".onnx"))

        class FakeExecutionMode:
            ORT_SEQUENTIAL = "ORT_SEQUENTIAL"
            ORT_PARALLEL = "ORT_PARALLEL"

        class FakeGraphOptimizationLevel:
            ORT_ENABLE_ALL = "ORT_ENABLE_ALL"
            ORT_DISABLE_ALL = "ORT_DISABLE_ALL"

        class FakeSessionOptions:
            def __init__(self):
                self.graph_optimization_level = None
                self.execution_mode = None
                self.intra_op_num_threads = None
                self.inter_op_num_threads = None

        monkeypatch.setattr(ModelManagerModule.onnxruntime, "SessionOptions", FakeSessionOptions)
        monkeypatch.setattr(ModelManagerModule.onnxruntime, "ExecutionMode", FakeExecutionMode)
        monkeypatch.setattr(ModelManagerModule.onnxruntime, "GraphOptimizationLevel", FakeGraphOptimizationLevel)

        def fake_inference_session(model_path, providers=None, provider_options=None, sess_options=None):
            created.append({
                "model_path": model_path,
                "providers": providers,
                "provider_options": provider_options,
                "sess_options": sess_options,
            })
            return object()

        monkeypatch.setattr(ModelManagerModule.onnxruntime, "InferenceSession", fake_inference_session)

        ok = manager.load_character(
            character_name="alice",
            model_dir="/tmp/model",
            language="English",
            runtime_config={
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "provider_options": {"CUDAExecutionProvider": {"device_id": "1"}},
                "execution_mode": "ORT_PARALLEL",
                "graph_optimization_level": "ORT_DISABLE_ALL",
                "intra_op_num_threads": 4,
                "inter_op_num_threads": 2,
            },
        )

        assert ok is True
        assert created
        assert all(item["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"] for item in created)
        assert all(item["provider_options"] == {"CUDAExecutionProvider": {"device_id": "1"}} for item in created)
        assert all(item["sess_options"].execution_mode == "ORT_PARALLEL" for item in created)
        assert all(item["sess_options"].graph_optimization_level == "ORT_DISABLE_ALL" for item in created)
        assert all(item["sess_options"].intra_op_num_threads == 4 for item in created)
        assert all(item["sess_options"].inter_op_num_threads == 2 for item in created)

    def test_load_character_accepts_runtime_config(self, monkeypatch):
        captured = {}

        monkeypatch.setattr(Internal, "check_onnx_model_dir", lambda _: None)
        monkeypatch.setattr(Internal, "ensure_exists", lambda *args, **kwargs: None)

        def fake_load_character(**kwargs):
            captured.update(kwargs)
            return True

        monkeypatch.setattr(Internal.model_manager, "load_character", fake_load_character)

        runtime_config = {
            "providers": ["CPUExecutionProvider"],
            "intra_op_num_threads": 2,
        }

        Internal.load_character(
            character_name="alice",
            onnx_model_dir="/tmp/model",
            language="en",
            runtime_config=runtime_config,
        )

        assert captured["runtime_config"]["providers"] == ["CPUExecutionProvider"]
        assert captured["runtime_config"]["intra_op_num_threads"] == 2

    def test_normalize_runtime_config_uses_environment_defaults(self, monkeypatch):
        monkeypatch.setenv("GENIE_ORT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
        monkeypatch.setenv("GENIE_ORT_INTRA_OP_NUM_THREADS", "6")
        monkeypatch.setenv("GENIE_ORT_INTER_OP_NUM_THREADS", "2")
        monkeypatch.setenv("GENIE_ORT_EXECUTION_MODE", "ORT_PARALLEL")
        monkeypatch.setenv("GENIE_ORT_GRAPH_OPTIMIZATION_LEVEL", "ORT_DISABLE_ALL")

        runtime = ModelManager.normalize_runtime_config(None)

        assert runtime.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert runtime.intra_op_num_threads == 6
        assert runtime.inter_op_num_threads == 2
        assert runtime.execution_mode == "ORT_PARALLEL"
        assert runtime.graph_optimization_level == "ORT_DISABLE_ALL"

    def test_normalize_runtime_config_explicit_values_override_environment(self, monkeypatch):
        monkeypatch.setenv("GENIE_ORT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
        monkeypatch.setenv("GENIE_ORT_INTRA_OP_NUM_THREADS", "6")
        monkeypatch.setenv("GENIE_ORT_INTER_OP_NUM_THREADS", "2")

        runtime = ModelManager.normalize_runtime_config({
            "providers": ["CPUExecutionProvider"],
            "intra_op_num_threads": 4,
            "inter_op_num_threads": 1,
        })

        assert runtime.providers == ["CPUExecutionProvider"]
        assert runtime.intra_op_num_threads == 4
        assert runtime.inter_op_num_threads == 1

    def test_start_server_uses_environment_defaults(self, monkeypatch):
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(Server.uvicorn, "run", fake_run)
        monkeypatch.setenv("GENIE_WORKERS", "3")
        monkeypatch.setenv("GENIE_MAX_CONCURRENCY", "5")
        monkeypatch.setenv("GENIE_QUEUE_MAXSIZE", "11")

        Server.start_server(host="127.0.0.1", port=8000)

        assert captured["workers"] == 3
        assert Server._server_runtime.single_process is False
        assert Server._server_runtime.max_concurrency == 5
        assert Server._server_runtime.queue_maxsize == 11

    def test_start_server_single_process_options(self, monkeypatch):
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(Server.uvicorn, "run", fake_run)

        Server.start_server(
            host="127.0.0.1",
            port=9000,
            workers=1,
            max_concurrency=3,
            queue_maxsize=9,
        )

        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 9000
        assert captured["workers"] == 1
        assert Server._server_runtime.max_concurrency == 3
        assert Server._server_runtime.queue_maxsize == 9
        assert Server._server_runtime.single_process is True

    def test_start_server_explicit_values_override_environment(self, monkeypatch):
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(Server.uvicorn, "run", fake_run)
        monkeypatch.setenv("GENIE_WORKERS", "3")
        monkeypatch.setenv("GENIE_MAX_CONCURRENCY", "5")
        monkeypatch.setenv("GENIE_QUEUE_MAXSIZE", "11")

        Server.start_server(
            host="127.0.0.1",
            port=8000,
            workers=2,
            max_concurrency=7,
            queue_maxsize=13,
        )

        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 8000
        assert captured["workers"] == 2
        assert Server._server_runtime.single_process is False
        assert Server._server_runtime.max_concurrency == 7
        assert Server._server_runtime.queue_maxsize == 13


    @pytest.mark.anyio("asyncio")
    async def test_multi_worker_acquire_does_not_touch_counters(self):
        Server._server_runtime.single_process = False
        Server._server_runtime.active_requests = 0
        Server._server_runtime.waiting_requests = 0

        await Server._acquire_tts_slot()

        assert Server._server_runtime.active_requests == 0
        assert Server._server_runtime.waiting_requests == 0

    @pytest.mark.anyio("asyncio")
    async def test_single_process_queue_limit_rejects_when_full(self, monkeypatch):
        class FakeHTTPException(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        monkeypatch.setattr(Server, "HTTPException", FakeHTTPException)
        Server._server_runtime.single_process = True
        Server._server_runtime.max_concurrency = 1
        Server._server_runtime.queue_maxsize = 1
        Server._server_runtime.active_requests = 1
        Server._server_runtime.waiting_requests = 1

        with pytest.raises(FakeHTTPException) as exc_info:
            await Server._acquire_tts_slot()

        assert exc_info.value.status_code == 429
    @pytest.mark.anyio("asyncio")
    async def test_single_process_waits_when_queue_has_capacity(self):
        Server._server_runtime.single_process = True
        Server._server_runtime.max_concurrency = 1
        Server._server_runtime.queue_maxsize = 1
        Server._server_runtime.active_requests = 1
        Server._server_runtime.waiting_requests = 0

        acquire_task = asyncio.create_task(Server._acquire_tts_slot())
        await asyncio.sleep(0)

        assert Server._server_runtime.waiting_requests == 1
        assert acquire_task.done() is False

        Server._release_tts_slot()
        await asyncio.wait_for(acquire_task, timeout=1)

        assert Server._server_runtime.active_requests == 1
        assert Server._server_runtime.waiting_requests == 0
        Server._release_tts_slot()
        assert Server._server_runtime.active_requests == 0

    @pytest.mark.anyio("asyncio")
    async def test_single_process_acquire_increments_active_requests(self):
        Server._server_runtime.single_process = True
        Server._server_runtime.max_concurrency = 2
        Server._server_runtime.queue_maxsize = 1
        Server._server_runtime.active_requests = 0
        Server._server_runtime.waiting_requests = 0

        await Server._acquire_tts_slot()

        assert Server._server_runtime.active_requests == 1
        Server._release_tts_slot()
        assert Server._server_runtime.active_requests == 0

    @pytest.mark.anyio("asyncio")
    async def test_tracked_audio_stream_releases_single_process_slot_on_finish(self):
        Server._server_runtime.single_process = True
        Server._server_runtime.max_concurrency = 1
        Server._server_runtime.queue_maxsize = 0
        Server._server_runtime.active_requests = 1
        Server._server_runtime.waiting_requests = 0

        queue = asyncio.Queue()
        await queue.put(None)

        chunks = []
        async for chunk in Server._tracked_audio_stream_generator(queue):
            chunks.append(chunk)

        assert chunks == []
        assert Server._server_runtime.active_requests == 0


class TestModelManagerRuntimeConfig:
    def test_load_roberta_uses_runtime_config_provider_options(self, monkeypatch):
        manager = ModelManager()
        manager.runtime_config = ModelManager.normalize_runtime_config({
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "provider_options": {"CUDAExecutionProvider": {"device_id": "2"}},
            "intra_op_num_threads": 3,
        })
        created = {}

        monkeypatch.setattr(os.path, "exists", lambda path: True)

        class FakeGraphOptimizationLevel:
            ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

        class FakeSessionOptions:
            def __init__(self):
                self.graph_optimization_level = None
                self.intra_op_num_threads = None
                self.inter_op_num_threads = None

        monkeypatch.setattr(ModelManagerModule.onnxruntime, "SessionOptions", FakeSessionOptions)
        monkeypatch.setattr(ModelManagerModule.onnxruntime, "GraphOptimizationLevel", FakeGraphOptimizationLevel)
        monkeypatch.setattr(ModelManagerModule.Tokenizer, "from_file", lambda path: object())

        def fake_inference_session(model_path, providers=None, provider_options=None, sess_options=None):
            created["providers"] = providers
            created["provider_options"] = provider_options
            created["sess_options"] = sess_options
            return object()

        monkeypatch.setattr(ModelManagerModule.onnxruntime, "InferenceSession", fake_inference_session)

        assert manager.load_roberta_model("/tmp/roberta.onnx") is True
        assert created["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert created["provider_options"] == {"CUDAExecutionProvider": {"device_id": "2"}}
        assert created["sess_options"].intra_op_num_threads == 3

    def test_normalize_runtime_config_preserves_provider_options(self):
        runtime = ModelManager.normalize_runtime_config({
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "provider_options": {"CUDAExecutionProvider": {"device_id": "1"}},
            "intra_op_num_threads": 4,
            "inter_op_num_threads": 2,
        })

        assert runtime.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert runtime.provider_options == {"CUDAExecutionProvider": {"device_id": "1"}}
        assert runtime.intra_op_num_threads == 4
        assert runtime.inter_op_num_threads == 2
