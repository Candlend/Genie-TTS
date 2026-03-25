import os
import pytest

from genie_tts import Internal, Server
from genie_tts.ModelManager import ModelManager


class TestRuntimeConfigAPI:
    def test_model_manager_uses_runtime_config_providers(self, monkeypatch):
        manager = ModelManager()
        created = []

        monkeypatch.setattr(os.path, "exists", lambda path: path.endswith(".onnx"))

        class FakeGraphOptimizationLevel:
            ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

        class FakeSessionOptions:
            def __init__(self):
                self.graph_optimization_level = None
                self.intra_op_num_threads = None
                self.inter_op_num_threads = None

        monkeypatch.setattr("genie_tts.ModelManager.onnxruntime.SessionOptions", FakeSessionOptions)
        monkeypatch.setattr("genie_tts.ModelManager.onnxruntime.GraphOptimizationLevel", FakeGraphOptimizationLevel)

        def fake_inference_session(model_path, providers=None, sess_options=None):
            created.append({
                "model_path": model_path,
                "providers": providers,
                "sess_options": sess_options,
            })
            return object()

        monkeypatch.setattr("genie_tts.ModelManager.onnxruntime.InferenceSession", fake_inference_session)

        ok = manager.load_character(
            character_name="alice",
            model_dir="/tmp/model",
            language="English",
            runtime_config={
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "intra_op_num_threads": 4,
                "inter_op_num_threads": 2,
            },
        )

        assert ok is True
        assert created
        assert all(item["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"] for item in created)
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

    def test_start_server_passes_workers_through(self, monkeypatch):
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(Server.uvicorn, "run", fake_run)

        Server.start_server(
            host="127.0.0.1",
            port=8000,
            workers=2,
        )

        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 8000
        assert captured["workers"] == 2

    def test_start_server_rejects_multi_worker_single_process_mode(self, monkeypatch):
        monkeypatch.setattr(Server.uvicorn, "run", lambda *args, **kwargs: None)

        with pytest.raises(ValueError, match="single-process"):
            Server.start_server(
                host="127.0.0.1",
                port=8000,
                workers=2,
                scaling_mode="single-process",
            )


class TestModelManagerRuntimeConfig:
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
