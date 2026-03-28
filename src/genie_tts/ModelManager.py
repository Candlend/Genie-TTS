"""
不再新建 .bin 文件。
修改后内存: 6448 MB
修改前内存: 5952 MB
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import onnx
import onnxruntime
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

from .Core.Resources import (HUBERT_MODEL_DIR, SV_MODEL, ROBERTA_MODEL_DIR)
from .Utils.Utils import LRUCacheDict

onnxruntime.set_default_logger_severity(3)
logger = logging.getLogger(__name__)


class GSVModelFile:
    T2S_ENCODER_FP32: str = 't2s_encoder_fp32.onnx'

    T2S_FIRST_STAGE_DECODER_FP32: str = 't2s_first_stage_decoder_fp32.onnx'
    T2S_FIRST_STAGE_DECODER_FP16: str = 't2s_first_stage_decoder_fp16.onnx'
    T2S_STAGE_DECODER_FP32: str = 't2s_stage_decoder_fp32.onnx'
    T2S_STAGE_DECODER_FP16: str = 't2s_stage_decoder_fp16.onnx'
    T2S_DECODER_WEIGHT_FP16: str = 't2s_shared_fp16.bin'

    VITS_FP32: str = 'vits_fp32.onnx'
    VITS_WEIGHT_FP16: str = 'vits_fp16.bin'

    PROMPT_ENCODER: str = 'prompt_encoder_fp32.onnx'
    PROMPT_ENCODER_WEIGHT_FP16: str = 'prompt_encoder_fp16.bin'

    HUBERT_MODEL = os.path.join(HUBERT_MODEL_DIR, "chinese-hubert-base.onnx")
    HUBERT_MODEL_WEIGHT_FP16 = os.path.join(HUBERT_MODEL_DIR, "chinese-hubert-base_weights_fp16.bin")

    ROBERTA_MODEL = os.path.join(ROBERTA_MODEL_DIR, 'RoBERTa.onnx')
    ROBERTA_TOKENIZER = os.path.join(ROBERTA_MODEL_DIR, 'roberta_tokenizer')


@dataclass
class RuntimeConfig:
    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    provider_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    intra_op_num_threads: Optional[int] = None
    inter_op_num_threads: Optional[int] = None
    execution_mode: Optional[str] = None
    graph_optimization_level: Optional[str] = "ORT_ENABLE_ALL"


@dataclass
class SessionCreateConfig:
    providers: List[str]
    provider_options: Optional[Dict[str, Dict[str, Any]]]
    sess_options: onnxruntime.SessionOptions


@dataclass
class GSVModel:
    LANGUAGE: str
    T2S_ENCODER: InferenceSession
    T2S_FIRST_STAGE_DECODER: InferenceSession
    T2S_STAGE_DECODER: InferenceSession
    VITS: InferenceSession
    PROMPT_ENCODER: Optional[InferenceSession] = None
    PROMPT_ENCODER_PATH: Optional[str] = None


def load_session_with_fp16_conversion(
        onnx_path: str,
        fp16_bin_path: str,
        providers: List[str],
        sess_options: Optional[onnxruntime.SessionOptions] = None,
        provider_options: Optional[Dict[str, Dict[str, Any]]] = None,
) -> InferenceSession:
    """
    通用函数：读取 ONNX 和 FP16 权重文件，在内存中将权重转换为 FP32，
    注入到 ONNX 模型中并加载 InferenceSession，不产生临时文件。
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX Model not found: {onnx_path}")
    if not os.path.exists(fp16_bin_path):
        raise FileNotFoundError(f"FP16 Weight file not found: {fp16_bin_path}")

    model_proto = onnx.load(onnx_path, load_external_data=False)
    fp16_data = np.fromfile(fp16_bin_path, dtype=np.float16)
    fp32_data = fp16_data.astype(np.float32)
    fp32_bytes = fp32_data.tobytes()

    # 遍历并修补模型中的 External Data Initializers
    for tensor in model_proto.graph.initializer:
        # 检查该 Tensor 是否使用外部数据
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            offset = 0
            length = 0
            # 解析外部数据信息
            for entry in tensor.external_data:
                if entry.key == 'offset':
                    offset = int(entry.value)
                elif entry.key == 'length':
                    length = int(entry.value)

            if offset + length > len(fp32_bytes):
                logger.warning(
                    f"Tensor {tensor.name} requested a data range that exceeds the size of the provided bin file. "
                    f"Offset: {offset}, Length: {length}, Buffer: {len(fp32_bytes)}"
                )
                continue

            tensor_data = fp32_bytes[offset: offset + length]
            tensor.raw_data = tensor_data

            del tensor.external_data[:]
            tensor.data_location = onnx.TensorProto.DEFAULT

    try:
        session = InferenceSession(
            model_proto.SerializeToString(),
            providers=providers,
            provider_options=provider_options,
            sess_options=sess_options
        )
        return session
    except Exception as e:
        logger.error(f"Failed to load in-memory model {os.path.basename(onnx_path)}: {e}")
        raise e


class ModelManager:
    def __init__(self):
        capacity_str = os.getenv('Max_Cached_Character_Models', '3')
        self.character_to_model: Dict[str, Dict[str, Optional[InferenceSession]]] = LRUCacheDict(
            capacity=int(capacity_str)
        )
        self.character_to_language: Dict[str, str] = {}
        self.character_model_paths: Dict[str, str] = {}
        self.runtime_config = RuntimeConfig()

        self.cn_hubert: Optional[InferenceSession] = None
        self.speaker_verification_model: Optional[InferenceSession] = None
        self.roberta_model: Optional[InferenceSession] = None
        self.roberta_tokenizer: Optional[Tokenizer] = None

    @staticmethod
    def normalize_runtime_config(runtime_config: Optional[Dict[str, Any] | RuntimeConfig]) -> RuntimeConfig:
        env_providers = os.getenv("GENIE_ORT_PROVIDERS")
        env_intra = os.getenv("GENIE_ORT_INTRA_OP_NUM_THREADS")
        env_inter = os.getenv("GENIE_ORT_INTER_OP_NUM_THREADS")
        env_execution_mode = os.getenv("GENIE_ORT_EXECUTION_MODE")
        env_graph_opt = os.getenv("GENIE_ORT_GRAPH_OPTIMIZATION_LEVEL")

        defaults = {
            "providers": [p.strip() for p in env_providers.split(",") if p.strip()] if env_providers else ["CPUExecutionProvider"],
            "provider_options": {},
            "intra_op_num_threads": int(env_intra) if env_intra else None,
            "inter_op_num_threads": int(env_inter) if env_inter else None,
            "execution_mode": env_execution_mode,
            "graph_optimization_level": env_graph_opt or "ORT_ENABLE_ALL",
        }

        if runtime_config is None:
            return RuntimeConfig(**defaults)
        if isinstance(runtime_config, RuntimeConfig):
            return runtime_config
        return RuntimeConfig(
            providers=list(runtime_config.get("providers", defaults["providers"])),
            provider_options=dict(runtime_config.get("provider_options", defaults["provider_options"])),
            intra_op_num_threads=runtime_config.get("intra_op_num_threads", defaults["intra_op_num_threads"]),
            inter_op_num_threads=runtime_config.get("inter_op_num_threads", defaults["inter_op_num_threads"]),
            execution_mode=runtime_config.get("execution_mode", defaults["execution_mode"]),
            graph_optimization_level=runtime_config.get("graph_optimization_level", defaults["graph_optimization_level"]),
        )

    @staticmethod
    def _build_session_create_config(runtime: RuntimeConfig) -> SessionCreateConfig:
        sess_options = onnxruntime.SessionOptions()
        graph_optimization_level = runtime.graph_optimization_level or "ORT_ENABLE_ALL"
        sess_options.graph_optimization_level = getattr(
            onnxruntime.GraphOptimizationLevel,
            graph_optimization_level,
        )
        if runtime.execution_mode is not None:
            sess_options.execution_mode = getattr(onnxruntime.ExecutionMode, runtime.execution_mode)
        if runtime.intra_op_num_threads is not None:
            sess_options.intra_op_num_threads = runtime.intra_op_num_threads
        if runtime.inter_op_num_threads is not None:
            sess_options.inter_op_num_threads = runtime.inter_op_num_threads

        provider_options = runtime.provider_options or None

        return SessionCreateConfig(
            providers=runtime.providers,
            provider_options=provider_options,
            sess_options=sess_options,
        )

    def load_roberta_model(self, model_path: str = GSVModelFile.ROBERTA_MODEL) -> bool:
        if self.roberta_model is not None:
            return True
        if not os.path.exists(model_path):
            # logger.warning(f'RoBERTa model does not exist: {model_path}. BERT features will not be used.')
            return False
        try:
            session_config = self._build_session_create_config(self.runtime_config)
            self.roberta_model = onnxruntime.InferenceSession(
                model_path,
                providers=session_config.providers,
                provider_options=session_config.provider_options,
                sess_options=session_config.sess_options,
            )
            self.roberta_tokenizer = Tokenizer.from_file(
                os.path.join(GSVModelFile.ROBERTA_TOKENIZER, 'tokenizer.json')
            )
            logger.info(f"Successfully loaded RoBERTa model.")
            return True
        except Exception as e:
            logger.error(
                f"Error: Failed to load ONNX model '{GSVModelFile.ROBERTA_MODEL}'.\n"
                f"Details: {e}"
            )
        return False

    def load_sv_model(self, model_path: str = SV_MODEL) -> bool:
        if self.speaker_verification_model is not None:
            return True
        try:
            session_config = self._build_session_create_config(self.runtime_config)
            self.speaker_verification_model = onnxruntime.InferenceSession(
                model_path,
                providers=session_config.providers,
                provider_options=session_config.provider_options,
                sess_options=session_config.sess_options,
            )
            logger.info(f"Successfully loaded Speaker Verification model.")
            return True
        except Exception as e:
            logger.error(
                f"Error: Failed to load ONNX model '{SV_MODEL}'.\n"
                f"Details: {e}"
            )
        return False

    def load_cn_hubert(self, model_path: str = GSVModelFile.HUBERT_MODEL) -> bool:
        if self.cn_hubert is not None:
            return True
        try:
            session_config = self._build_session_create_config(self.runtime_config)
            # Hubert 也应用内存转换逻辑
            if model_path == GSVModelFile.HUBERT_MODEL and os.path.exists(GSVModelFile.HUBERT_MODEL_WEIGHT_FP16):
                self.cn_hubert = load_session_with_fp16_conversion(
                    model_path,
                    GSVModelFile.HUBERT_MODEL_WEIGHT_FP16,
                    session_config.providers,
                    session_config.sess_options,
                    session_config.provider_options,
                )
            else:
                self.cn_hubert = onnxruntime.InferenceSession(
                    model_path,
                    providers=session_config.providers,
                    provider_options=session_config.provider_options,
                    sess_options=session_config.sess_options,
                )
            logger.info("Successfully loaded CN_HuBERT model.")
            return True
        except Exception as e:
            logger.error(
                f"Error: Failed to load ONNX model '{GSVModelFile.HUBERT_MODEL}'.\n"
                f"Details: {e}"
            )
        return False

    def get(self, character_name: str) -> Optional[GSVModel]:
        character_name = character_name.lower()
        language = self.character_to_language.get(character_name, 'Japanese')
        if character_name in self.character_to_model:
            model_map: dict = self.character_to_model[character_name]
            # 简化获取逻辑
            t2s_first_stage_decoder = model_map.get(GSVModelFile.T2S_FIRST_STAGE_DECODER_FP32) or \
                                      model_map.get(GSVModelFile.T2S_FIRST_STAGE_DECODER_FP16)
            t2s_stage_decoder = model_map.get(GSVModelFile.T2S_STAGE_DECODER_FP32) or \
                                model_map.get(GSVModelFile.T2S_STAGE_DECODER_FP16)
            prompt_encoder_path = os.path.join(self.character_model_paths[character_name], GSVModelFile.PROMPT_ENCODER)

            return GSVModel(
                LANGUAGE=language,
                T2S_ENCODER=model_map[GSVModelFile.T2S_ENCODER_FP32],
                T2S_FIRST_STAGE_DECODER=t2s_first_stage_decoder,
                T2S_STAGE_DECODER=t2s_stage_decoder,
                VITS=model_map[GSVModelFile.VITS_FP32],
                PROMPT_ENCODER=model_map[GSVModelFile.PROMPT_ENCODER],
                PROMPT_ENCODER_PATH=prompt_encoder_path,
            )
        if character_name in self.character_model_paths:
            model_dir = self.character_model_paths[character_name]
            if self.load_character(character_name, model_dir, language=language):
                return self.get(character_name)
            else:
                del self.character_model_paths[character_name]
                return None
        return None

    def has_character(self, character_name: str) -> bool:
        character_name = character_name.lower()
        return character_name in self.character_model_paths

    def load_character(
            self,
            character_name: str,
            model_dir: str,
            language: str,
            runtime_config: Optional[Dict[str, Any] | RuntimeConfig] = None,
    ) -> bool:
        """
        加载角色模型，如果需要，在内存中动态转换 FP16 权重。
        """
        character_name = character_name.lower()
        runtime = self.normalize_runtime_config(runtime_config)
        if character_name in self.character_to_model:
            _ = self.character_to_model[character_name]
            return True

        model_dict: Dict[str, Optional[InferenceSession]] = {}

        # 定义 ONNX 文件到 FP16 Bin 文件的映射关系
        onnx_to_fp16_map = {
            GSVModelFile.T2S_FIRST_STAGE_DECODER_FP32: GSVModelFile.T2S_DECODER_WEIGHT_FP16,
            GSVModelFile.T2S_STAGE_DECODER_FP32: GSVModelFile.T2S_DECODER_WEIGHT_FP16,
            GSVModelFile.VITS_FP32: GSVModelFile.VITS_WEIGHT_FP16,
            GSVModelFile.PROMPT_ENCODER: GSVModelFile.PROMPT_ENCODER_WEIGHT_FP16
        }

        # 确定需要加载的模型列表
        model_files_to_load = [
            GSVModelFile.T2S_ENCODER_FP32,
            GSVModelFile.VITS_FP32,
            GSVModelFile.PROMPT_ENCODER,
        ]

        fp32_decoders = [GSVModelFile.T2S_FIRST_STAGE_DECODER_FP32, GSVModelFile.T2S_STAGE_DECODER_FP32]
        model_files_to_load.extend(fp32_decoders)

        try:
            session_config = self._build_session_create_config(runtime)
            for model_file in model_files_to_load:
                model_path = os.path.normpath(os.path.join(model_dir, model_file))

                if os.path.exists(model_path):
                    fp16_bin_name = onnx_to_fp16_map.get(model_file)
                    fp16_bin_path = os.path.join(model_dir, fp16_bin_name) if fp16_bin_name else None

                    if fp16_bin_path and os.path.exists(fp16_bin_path):
                        model_dict[model_file] = load_session_with_fp16_conversion(
                            model_path,
                            fp16_bin_path,
                            session_config.providers,
                            session_config.sess_options,
                            session_config.provider_options,
                        )
                    else:
                        model_dict[model_file] = onnxruntime.InferenceSession(
                            model_path,
                            providers=session_config.providers,
                            provider_options=session_config.provider_options,
                            sess_options=session_config.sess_options,
                        )
                elif model_file == GSVModelFile.PROMPT_ENCODER:
                    model_dict[model_file] = None
                else:
                    raise FileNotFoundError(f'文件 {model_path} 不存在！')

            # 日志信息
            is_v2pp = model_dict[GSVModelFile.PROMPT_ENCODER] is not None
            logger.info(
                f"Character {character_name.capitalize()} loaded successfully.\n"
                f"- Model Path: {model_dir}\n"
                f"- Model Type: {'V2ProPlus' if is_v2pp else 'V2'}"
            )

            self.character_to_model[character_name] = model_dict
            self.character_to_language[character_name] = language
            self.character_model_paths[character_name] = model_dir
            return True

        except Exception as e:
            logger.error(
                f"Error: Failed to load ONNX model '{model_dir}'.\n"
                f"Details: {e}"
            )
            return False

    def remove_all_character(self) -> None:
        self.character_to_model.clear()
        gc.collect()

    def remove_character(self, character_name: str) -> None:
        character_name = character_name.lower()
        if character_name in self.character_to_model:
            del self.character_to_model[character_name]
            gc.collect()
            logger.info(f"Character {character_name.capitalize()} removed successfully.")


model_manager: ModelManager = ModelManager()
