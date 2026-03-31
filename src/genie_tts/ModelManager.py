import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import onnx
import onnxruntime
from onnxruntime import InferenceSession
from onnxruntime.transformers.float16 import convert_float_to_float16
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
        _fp32_bytes_cache: Optional[bytes] = None,
        convert_graph: bool = False,
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
    if _fp32_bytes_cache is not None:
        fp32_bytes = _fp32_bytes_cache
    else:
        fp16_data = np.fromfile(fp16_bin_path, dtype=np.float16)
        fp32_data = fp16_data.astype(np.float32)
        fp32_bytes = fp32_data.tobytes()
        del fp16_data, fp32_data

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

    if convert_graph and "CUDAExecutionProvider" in providers:
        model_proto = convert_float_to_float16(
            model_proto,
            keep_io_types=True,
            disable_shape_infer=True,
        )
        # 清空中间值的静态 shape 信息，防止 ORT 按错误的静态 shape 预分配 buffer
        # 对自回归模型（KV cache 每步增长）尤其关键
        del model_proto.graph.value_info[:]

    try:
        session = InferenceSession(
            model_proto.SerializeToString(),
            providers=providers,
            sess_options=sess_options
        )
        del model_proto
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
        available = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in available:
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Using CUDAExecutionProvider for inference.")
        else:
            self.providers = ["CPUExecutionProvider"]
            logger.info("CUDAExecutionProvider not available, falling back to CPU.")

        self.cn_hubert: Optional[InferenceSession] = None
        self.speaker_verification_model: Optional[InferenceSession] = None
        self.roberta_model: Optional[InferenceSession] = None
        self.roberta_tokenizer: Optional[Tokenizer] = None

    def load_roberta_model(self, model_path: str = GSVModelFile.ROBERTA_MODEL) -> bool:
        if self.roberta_model is not None:
            return True
        if not os.path.exists(model_path):
            # logger.warning(f'RoBERTa model does not exist: {model_path}. BERT features will not be used.')
            return False
        try:
            _opts = onnxruntime.SessionOptions()
            _opts.enable_cpu_mem_arena = False
            self.roberta_model = onnxruntime.InferenceSession(
                model_path,
                providers=self.providers,
                sess_options=_opts,
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
            _opts = onnxruntime.SessionOptions()
            _opts.enable_cpu_mem_arena = False
            self.speaker_verification_model = onnxruntime.InferenceSession(
                model_path,
                providers=self.providers,
                sess_options=_opts,
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
            # Hubert 也应用内存转换逻辑
            _opts = onnxruntime.SessionOptions()
            _opts.enable_cpu_mem_arena = False
            if model_path == GSVModelFile.HUBERT_MODEL and os.path.exists(GSVModelFile.HUBERT_MODEL_WEIGHT_FP16):
                self.cn_hubert = load_session_with_fp16_conversion(
                    model_path,
                    GSVModelFile.HUBERT_MODEL_WEIGHT_FP16,
                    self.providers,
                    _opts,
                    convert_graph=True,
                )
            else:
                self.cn_hubert = onnxruntime.InferenceSession(
                    model_path,
                    providers=self.providers,
                    sess_options=_opts,
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
            t2s_first_stage_decoder: InferenceSession = (
                model_map.get(GSVModelFile.T2S_FIRST_STAGE_DECODER_FP32) or
                model_map.get(GSVModelFile.T2S_FIRST_STAGE_DECODER_FP16)  # type: ignore[assignment]
            )
            t2s_stage_decoder: InferenceSession = (
                model_map.get(GSVModelFile.T2S_STAGE_DECODER_FP32) or
                model_map.get(GSVModelFile.T2S_STAGE_DECODER_FP16)  # type: ignore[assignment]
            )
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
    ) -> bool:
        """
        加载角色模型，如果需要，在内存中动态转换 FP16 权重。
        """
        character_name = character_name.lower()
        if character_name in self.character_to_model:
            _ = self.character_to_model[character_name]
            return True

        model_dict: Dict[str, Optional[InferenceSession]] = {}

        # 非自回归模型可以安全做图 fp16 转换（无 KV cache 动态 shape 问题）
        # T2S encoder 没有 fp16 bin，走 else 分支；VITS/PROMPT_ENCODER 有 fp16 bin，走 load_session_with_fp16_conversion
        _GRAPH_FP16_SAFE = {
            GSVModelFile.T2S_ENCODER_FP32,
            GSVModelFile.VITS_FP32,
            GSVModelFile.PROMPT_ENCODER,
        }

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

        # 预先展开 T2S 共享 fp16 bin，两个 decoder session 复用，避免读取两次
        _t2s_fp16_bin_path = os.path.normpath(os.path.join(model_dir, GSVModelFile.T2S_DECODER_WEIGHT_FP16))
        _t2s_fp32_bytes_cache: Optional[bytes] = None
        if os.path.exists(_t2s_fp16_bin_path):
            _fp16 = np.fromfile(_t2s_fp16_bin_path, dtype=np.float16)
            _t2s_fp32_bytes_cache = _fp16.astype(np.float32).tobytes()
            del _fp16

        try:
            for model_file in model_files_to_load:
                model_path = os.path.normpath(os.path.join(model_dir, model_file))

                # 设置 Session Options
                sess_options = onnxruntime.SessionOptions()
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.enable_cpu_mem_arena = False
                if "CUDAExecutionProvider" in self.providers:
                    sess_options.enable_mem_pattern = False

                if os.path.exists(model_path):
                    fp16_bin_name = onnx_to_fp16_map.get(model_file)
                    fp16_bin_path = os.path.join(model_dir, fp16_bin_name) if fp16_bin_name else None

                    if fp16_bin_path and os.path.exists(fp16_bin_path):
                        # T2S 两个 decoder 共享同一个 bin，复用已展开的 fp32_bytes 避免重复读取
                        cache = _t2s_fp32_bytes_cache if fp16_bin_path == _t2s_fp16_bin_path else None
                        model_dict[model_file] = load_session_with_fp16_conversion(
                            model_path, fp16_bin_path, self.providers, sess_options, cache,
                            convert_graph=True,
                        )
                    else:
                        if "CUDAExecutionProvider" in self.providers and model_file in _GRAPH_FP16_SAFE:
                            _proto = onnx.load(model_path)
                            _proto = convert_float_to_float16(
                                _proto,
                                keep_io_types=True,
                                disable_shape_infer=True,
                            )
                            model_dict[model_file] = onnxruntime.InferenceSession(
                                _proto.SerializeToString(),
                                providers=self.providers,
                                sess_options=sess_options,
                            )
                            del _proto
                        else:
                            model_dict[model_file] = onnxruntime.InferenceSession(
                                model_path,
                                providers=self.providers,
                                sess_options=sess_options,
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
