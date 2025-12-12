from pathlib import Path
import json
from typing import List

class HotwordStore:
    def __init__(self, file_path: str = "./data/hotwords.json"):
        self.file_path = Path(file_path)

    def _read_hotwords(self) -> List[str]:
        if not self.file_path.exists():
            return []
        text = self.file_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "hotwords" in data:
                return [str(w).strip() for w in data["hotwords"] if str(w).strip()]
            if isinstance(data, list):
                return [str(w).strip() for w in data if str(w).strip()]
        except json.JSONDecodeError:
            pass
        return [w for w in text.replace("\n", " ").split(" ") if w.strip()]

    def get_hotword_string(self) -> str:
        words = self._read_hotwords()
        if not words:
            return ""
        # FunASR 那边支持空格分隔
        return " ".join(words)

import time
import os
import sys
import io
import psutil
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.providers.asr.base import ASRProviderBase
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import shutil
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()

MAX_RETRIES = 2
RETRY_DELAY = 1  # 重试延迟（秒）

# https://www.modelscope.cn/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary

# 捕获标准输出
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()

        # 将捕获到的内容通过 logger 输出
        if self.output:
            logger.bind(tag=TAG).info(self.output.strip())


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        
        # 内存检测，要求大于2G
        min_mem_bytes = 2 * 1024 * 1024 * 1024
        total_mem = psutil.virtual_memory().total
        if total_mem < min_mem_bytes:
            logger.bind(tag=TAG).error(f"可用内存不足2G，当前仅有 {total_mem / (1024*1024):.2f} MB，可能无法启动FunASR")
        
        self.interface_type = InterfaceType.LOCAL
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")  # 修正配置键名
        self.delete_audio_file = delete_audio_file
        hotword_file = "./data/hotwords.json"
        self.hotword_store = HotwordStore(hotword_file)

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        with CaptureOutput():
            self.model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  device="cuda:0",  # 启用GPU加速
                #   device="cpu",  # 用cpu也能跑
                  )
            # self.model = AutoModel(model="dengcunqin/speech_seaco_paraformer_large_asr_nat-zh-cantonese-en-16k-common-vocab11666-pytorch",
            #       model_revision="master",device="cuda:0", hub="ms"
            #       )

            # AutoModel(
            #     model=self.model_dir,
            #     vad_kwargs={"max_single_segment_time": 30000},
            #     disable_update=True,
            #     hub="hf",
            #     device="cuda:0",  # 启用GPU加速
            # )

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        """语音转文本主处理逻辑"""
        file_path = None
        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                # 合并所有opus数据包
                if audio_format == "pcm":
                    pcm_data = opus_data
                else:
                    pcm_data = self.decode_opus(opus_data)

                combined_pcm_data = b"".join(pcm_data)

                # 检查磁盘空间
                if not self.delete_audio_file:
                    free_space = shutil.disk_usage(self.output_dir).free
                    if free_space < len(combined_pcm_data) * 2:  # 预留2倍空间
                        raise OSError("磁盘空间不足")

                # 判断是否保存为WAV文件
                if self.delete_audio_file:
                    pass
                else:
                    file_path = self.save_audio_to_file(pcm_data, session_id)

                # 语音识别
                start_time = time.time()
                hotword = self.hotword_store.get_hotword_string()
                logger.bind(tag=TAG).info(f"[ASR Hotword] hotword_str = {repr(hotword)}")
                result = self.model.generate(
                    input=combined_pcm_data,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    hotword=hotword, 
                )
                text = rich_transcription_postprocess(result[0]["text"])
                logger.bind(tag=TAG).debug(
                    f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}"
                )

                return text, file_path

            except OSError as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    logger.bind(tag=TAG).error(
                        f"语音识别失败（已重试{retry_count}次）: {e}", exc_info=True
                    )
                    return "", file_path
                logger.bind(tag=TAG).warning(
                    f"语音识别失败，正在重试（{retry_count}/{MAX_RETRIES}）: {e}"
                )
                time.sleep(RETRY_DELAY)

            except Exception as e:
                logger.bind(tag=TAG).error(f"语音识别失败: {e}", exc_info=True)
                return "", file_path

            finally:
                # 文件清理逻辑
                if self.delete_audio_file and file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.bind(tag=TAG).debug(f"已删除临时音频文件: {file_path}")
                    except Exception as e:
                        logger.bind(tag=TAG).error(
                            f"文件删除失败: {file_path} | 错误: {e}"
                        )
'''
注: FunASR也支持带词库的和流式的paraformer-zh-streaming 
但是不一定识别出感情
不过由于现有的vad+单句+识别, 本身的速度已经很快了, 也没太多必要用流式的了
用流式主要是那种带屏幕的, 实时显示字幕的场景(也许没这个需求)

https://github.com/modelscope/FunASR
Speech Recognition (Streaming)
from funasr import AutoModel

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming")

import soundfile
import os

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960 # 600ms

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
    print(res)
Note: chunk_size is the configuration for streaming latency. [0,10,5] indicates that the real-time display granularity is 10*60=600ms, and the lookahead information is 5*60=300ms. Each inference input is 600ms (sample points are 16000*0.6=960), and the output is the corresponding text. For the last speech segment input, is_final=True needs to be set to force the output of the last word.
'''

# from pathlib import Path
# import json
# from typing import List, Optional, Tuple
# import time
# import os
# import sys
# import io
# import psutil
# import shutil

# from config.logger import setup_logging
# from core.providers.asr.base import ASRProviderBase
# from core.providers.asr.dto.dto import InterfaceType

# from modelscope import snapshot_download

# TAG = __name__
# logger = setup_logging()

# MAX_RETRIES = 2
# RETRY_DELAY = 1  # 重试延迟（秒）

# # ======================
# # 热词存储
# # ======================
# class HotwordStore:
#     def __init__(self, file_path: str = "./data/hotwords.json"):
#         self.file_path = Path(file_path)

#     def _read_hotwords(self) -> List[str]:
#         if not self.file_path.exists():
#             return []
#         text = self.file_path.read_text(encoding="utf-8").strip()
#         if not text:
#             return []
#         try:
#             data = json.loads(text)
#             if isinstance(data, dict) and "hotwords" in data:
#                 return [str(w).strip() for w in data["hotwords"] if str(w).strip()]
#             if isinstance(data, list):
#                 return [str(w).strip() for w in data if str(w).strip()]
#         except json.JSONDecodeError:
#             pass
#         return [w for w in text.replace("\n", " ").split(" ") if w.strip()]

#     def get_hotword_string(self) -> str:
#         words = self._read_hotwords()
#         if not words:
#             return ""
#         # SenseVoice / FunASR 都支持空格分隔
#         return " ".join(words)


# # ======================
# # 捕获标准输出
# # ======================
# class CaptureOutput:
#     def __enter__(self):
#         self._output = io.StringIO()
#         self._original_stdout = sys.stdout
#         sys.stdout = self._output

#     def __exit__(self, exc_type, exc_value, traceback):
#         sys.stdout = self._original_stdout
#         self.output = self._output.getvalue()
#         self._output.close()

#         # 将捕获到的内容通过 logger 输出
#         if self.output:
#             logger.bind(tag=TAG).info(self.output.strip())


# # ======================
# # 下载并加载热词模型（模块级一次）
# # ======================
# MODEL_ID = "dengcunqin/SenseVoiceSmall_hotword"
# MODEL_DIR = snapshot_download(MODEL_ID)
# sys.path.append(MODEL_DIR)

# from sensevoice_bin_hot import SenseVoiceSmall
# from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess


# class ASRProvider(ASRProviderBase):
#     def __init__(self, config: dict, delete_audio_file: bool):
#         super().__init__()

#         # 内存检测，要求大于 2G
#         min_mem_bytes = 2 * 1024 * 1024 * 1024
#         total_mem = psutil.virtual_memory().total
#         if total_mem < min_mem_bytes:
#             logger.bind(tag=TAG).error(
#                 f"可用内存不足2G，当前仅有 {total_mem / (1024*1024):.2f} MB，可能无法启动 SenseVoiceSmall"
#             )

#         self.interface_type = InterfaceType.LOCAL
#         self.model_dir = config.get("model_dir")
#         self.output_dir = config.get("output_dir")
#         self.delete_audio_file = delete_audio_file
#         hotword_file = "./data/hotwords.json"
#         self.hotword_store = HotwordStore(hotword_file)

#         # 确保输出目录存在
#         os.makedirs(self.output_dir, exist_ok=True)

#         # 加载 ONNX 热词模型
#         with CaptureOutput():
#             self.model = SenseVoiceSmall(
#                 MODEL_DIR,
#                 batch_size=10,
#                 quantize=False,  # 你要省点算力也可以改 True
#             )

#     async def speech_to_text(
#         self, opus_data: List[bytes], session_id: str, audio_format: str = "opus"
#     ) -> Tuple[Optional[str], Optional[str]]:
#         """语音转文本主处理逻辑"""
#         file_path: Optional[str] = None
#         retry_count = 0

#         while retry_count < MAX_RETRIES:
#             try:
#                 # 合并所有 opus 数据包 / 直接使用 PCM
#                 if audio_format == "pcm":
#                     pcm_data = opus_data
#                 else:
#                     pcm_data = self.decode_opus(opus_data)

#                 combined_pcm_data = b"".join(pcm_data)

#                 # 检查磁盘空间（粗略 2 倍冗余）
#                 if not self.delete_audio_file:
#                     free_space = shutil.disk_usage(self.output_dir).free
#                     if free_space < len(combined_pcm_data) * 2:
#                         raise OSError("磁盘空间不足")

#                 # ⭐ 无论是否需要最终删除，这里都先保存文件，供 SenseVoiceSmall 使用
#                 file_path = self.save_audio_to_file(pcm_data, session_id)

#                 # 语音识别
#                 start_time = time.time()
#                 hotword = self.hotword_store.get_hotword_string()
#                 logger.bind(tag=TAG).info(
#                     f"[ASR Hotword] hotword_str = {repr(hotword)}"
#                 )

#                 # SenseVoiceSmall ONNX 调用
#                 # 官方 demo 是：res = model([wav_path], hotwords_str='打磨院', hotwords_score=1.0)
#                 wav_list = [file_path]
#                 if hotword:
#                     logger.bind(tag=TAG).info("using hot word")
#                     res = self.model(
#                         wav_list,
#                         language="auto",
#                         use_itn=True,
#                         hotwords_str=hotword,
#                         hotwords_score=1.0,
#                     )
#                 else:
#                     logger.bind(tag=TAG).info("no hot word")
#                     res = self.model(
#                         wav_list,
#                         language="auto",
#                         use_itn=True,
#                         hotwords_str="",
#                         hotwords_score=1.0,
#                     )

#                 # ONNX 版 demo: [rich_transcription_postprocess(i) for i in res]
#                 text = rich_transcription_postprocess(res[0])

#                 logger.bind(tag=TAG).debug(
#                     f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}"
#                 )

#                 return text, file_path

#             except OSError as e:
#                 retry_count += 1
#                 if retry_count >= MAX_RETRIES:
#                     logger.bind(tag=TAG).error(
#                         f"语音识别失败（已重试{retry_count}次）: {e}", exc_info=True
#                     )
#                     return "", file_path
#                 logger.bind(tag=TAG).warning(
#                     f"语音识别失败，正在重试（{retry_count}/{MAX_RETRIES}）: {e}"
#                 )
#                 time.sleep(RETRY_DELAY)

#             except Exception as e:
#                 retry_count += 1
#                 logger.bind(tag=TAG).error(
#                     f"语音识别失败（第 {retry_count} 次）: {e}", exc_info=True
#                 )
#                 if retry_count >= MAX_RETRIES:
#                     return "", file_path
#                 time.sleep(RETRY_DELAY)

#             finally:
#                 # 文件清理逻辑：如果配置要求删除，识别结束后删掉
#                 if self.delete_audio_file and file_path and os.path.exists(file_path):
#                     try:
#                         os.remove(file_path)
#                         logger.bind(tag=TAG).debug(f"已删除临时音频文件: {file_path}")
#                     except Exception as e:
#                         logger.bind(tag=TAG).error(
#                             f"文件删除失败: {file_path} | 错误: {e}"
#                         )
