import os
import traceback

from ..base import MemoryProviderBase, logger
from mem0 import Memory
from core.utils.util import check_model_key

TAG = __name__


class MemoryProvider(MemoryProviderBase):
    def __init__(self, config, summary_memory=None):
        super().__init__(config)

        # 兼容：保留原来的字段，避免外面有地方访问 self.api_key / self.api_version
        self.api_key = config.get("api_key", os.environ.get("QWEN_API_KEY", ""))
        self.api_version = config.get("api_version", "v1.1")

        # 仍然用原来的检查逻辑，只是现在这个 key 是给本地 LLM 用的
        model_key_msg = check_model_key("Mem0-Local", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)
            self.use_mem0 = False
            return
        else:
            self.use_mem0 = True

        # 1. 准备本地 mem0 的配置
        # 如果外部 config 里已经给了 mem0_config，就用外部的；
        # 否则就用你在 demo 里那套默认配置。
        mem0_config = config.get("mem0_config")
        if mem0_config is None:
            mem0_config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        # 对应你 demo 中的 dashscope 兼容模式
                        "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key": self.api_key,
                        "model": "qwen-turbo",
                        "temperature": 0.2,
                        "max_tokens": 2000,
                    }
                },

                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        # 和你本地脚本一样
                        "path": "./chroma_data",
                        "collection_name": "memories",
                    }
                },

                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": "nomic-embed-text",
                        "embedding_dims": 512,
                        "ollama_base_url": "http://localhost:11434",
                    }
                },
            }

        try:
            # 2. 用本地配置初始化 mem0
            self.client = Memory.from_config(mem0_config)
            logger.bind(tag=TAG).info("使用本地 Mem0 配置初始化成功")
        except Exception as e:
            logger.bind(tag=TAG).error(f"初始化本地 Mem0 配置失败: {str(e)}")
            logger.bind(tag=TAG).error(f"详细错误: {traceback.format_exc()}")
            self.use_mem0 = False

    async def save_memory(self, msgs):
        if not self.use_mem0:
            return None
        if len(msgs) < 2:
            return None

        try:
            # 保持和原来一样：过滤掉 system，把消息列表塞进 mem0
            messages = [
                {"role": message.role, "content": message.content}
                for message in msgs
                if message.role != "system"
            ]
            result = self.client.add(
                messages, user_id=self.role_id
            )
            logger.bind(tag=TAG).debug(f"Save memory result: {result}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存记忆失败: {str(e)}")
            return None

    async def query_memory(self, query: str) -> str:
        if not self.use_mem0:
            return ""
        try:
            if not getattr(self, "role_id", None):
                return ""

            # ✅ 这里是一个关键改动：
            # - 原来用的是 self.client.search(query, filters={"user_id": ...})
            # - 本地 Memory 的用法是 search(query=..., user_id=...)
            results = self.client.search(query=query, user_id=self.role_id)

            if not results:
                return ""

            # 兼容两种返回格式：
            # - 云端 client 版本：{"results": [...]}
            # - OSS 版本常见：{"memories": [...], "entities": [...]}
            raw_memories = []
            if "results" in results:
                raw_memories = results["results"]
            elif "memories" in results:
                raw_memories = results["memories"]

            if not raw_memories:
                return ""

            # 保持你原来的逻辑：按时间排序，格式化时间
            memories = []
            for entry in raw_memories:
                # 尽量兼容 updated_at / created_at
                timestamp = entry.get("updated_at") or entry.get("created_at") or ""
                if timestamp:
                    try:
                        dt = timestamp.split(".")[0]  # 去掉毫秒
                        formatted_time = dt.replace("T", " ")
                    except Exception:
                        formatted_time = timestamp
                else:
                    formatted_time = ""

                memory_text = entry.get("memory", "")
                if formatted_time and memory_text:
                    memories.append(
                        (timestamp, f"[{formatted_time}] {memory_text}")
                    )

            # 按时间倒序（最新在前）
            memories.sort(key=lambda x: x[0], reverse=True)

            memories_str = "\n".join(f"- {m[1]}" for m in memories)
            logger.bind(tag=TAG).debug(f"Query results: {memories_str}")
            return memories_str
        except Exception as e:
            logger.bind(tag=TAG).error(f"查询记忆失败: {str(e)}")
            return ""
