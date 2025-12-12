from aiohttp import web
from pathlib import Path
import asyncio
import json
from typing import List


class HotwordHandler:
    def __init__(self, config: dict):
        server_conf = config.get("server", {})
        # 统一从 config 里拿路径，ASR 那边也用这个路径
        hotword_file = server_conf.get("hotword_file", "./data/hotwords.json")

        self.file_path = Path(hotword_file)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def _read_hotwords(self) -> List[str]:
        if not self.file_path.exists():
            return []

        text = self.file_path.read_text(encoding="utf-8").strip()
        if not text:
            return []

        # 优先 JSON 格式：{"hotwords": ["左近前灯", "大堂灯1"]}
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "hotwords" in data:
                return [str(w).strip() for w in data["hotwords"] if str(w).strip()]
            if isinstance(data, list):
                return [str(w).strip() for w in data if str(w).strip()]
        except json.JSONDecodeError:
            pass

        # 兜底：把文件当成空格/换行分隔的字符串
        return [w for w in text.replace("\n", " ").split(" ") if w.strip()]

    async def _write_hotwords(self, words: List[str]):
        cleaned = [w.strip() for w in words if w and w.strip()]
        data = {"hotwords": cleaned}
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def handle_get(self, request: web.Request):
        """获取当前热词列表（JSON）"""
        async with self._lock:
            words = await self._read_hotwords()
        return web.json_response({"hotwords": words})

    async def handle_post(self, request: web.Request):
        """
        设置热词列表（覆盖式，JSON）
        body 可以是：
        {
          "hotwords": ["左近前灯", "大堂灯1", "水云间"]
        }
        或者：
        {
          "hotwords": "左近前灯 大堂灯1 水云间"
        }
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        words = data.get("hotwords") or data.get("words") or []

        if isinstance(words, str):
            # 支持前端直接传一整行字符串
            words = [w for w in words.replace("\n", " ").split(" ") if w.strip()]
        elif isinstance(words, list):
            words = [str(w) for w in words]
        else:
            return web.json_response(
                {"error": "hotwords must be list or string"},
                status=400,
            )

        async with self._lock:
            await self._write_hotwords(words)

        return web.json_response({"hotwords": words})

    async def handle_page(self, request: web.Request):
        """
        返回一个简单的 HTML 页面，用浏览器管理热词
        GET /mcp/asr/hotwords/page
        """
        html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8" />
    <title>热词词库配置</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
        body {
            font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
                         "Helvetica Neue",Arial,"Noto Sans","Liberation Sans",
                         "PingFang SC","Hiragino Sans GB","Microsoft YaHei",
                         "Source Han Sans SC",sans-serif;
            margin: 0;
            padding: 16px;
            background: #f5f5f5;
        }
        .container {
            max-width: 720px;
            margin: 0 auto;
            background: #ffffff;
            padding: 16px 20px 20px;
            border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        h1 {
            font-size: 20px;
            margin: 0 0 12px;
        }
        p {
            margin: 4px 0 10px;
            color: #666;
            font-size: 13px;
        }
        textarea {
            width: 100%;
            min-height: 220px;
            box-sizing: border-box;
            font-family: inherit;
            font-size: 14px;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #d0d0d0;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #409eff;
            box-shadow: 0 0 0 1px rgba(64,158,255,0.15);
        }
        .toolbar {
            margin-top: 10px;
            display: flex;
            gap: 8px;
            align-items: center;
        }
        button {
            padding: 6px 14px;
            border-radius: 4px;
            border: none;
            cursor: pointer;
            font-size: 14px;
        }
        #btn-save {
            background-color: #409eff;
            color: #fff;
        }
        #btn-reload {
            background-color: #e0e0e0;
        }
        #status {
            margin-left: 8px;
            font-size: 13px;
        }
        #status.ok {
            color: #2e7d32;
        }
        #status.error {
            color: #d32f2f;
        }
        .tip {
            font-size: 12px;
            color: #999;
            margin-top: 4px;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>热词词库配置</h1>
    <p>每行一个热词，例如：<code>左近前灯</code>、<code>大堂灯1</code>、<code>水云间</code>。</p>
    <textarea id="hotwords" placeholder="在这里输入热词，每行一个"></textarea>
    <div class="tip">保存后会写入服务器本地文件，ASR 识别前会自动读取并作为 hotword 传入。</div>
    <div class="toolbar">
        <button id="btn-save">保存</button>
        <button id="btn-reload">重新加载</button>
        <span id="status"></span>
    </div>
</div>

<script>
    const API_URL = "/mcp/asr/hotwords";

    function setStatus(text, type) {
        const el = document.getElementById("status");
        el.textContent = text || "";
        el.className = type ? type : "";
    }

    async function loadHotwords() {
        setStatus("加载中...", "");
        try {
            const res = await fetch(API_URL, { method: "GET" });
            if (!res.ok) {
                throw new Error("HTTP " + res.status);
            }
            const data = await res.json();
            const list = Array.isArray(data.hotwords) ? data.hotwords : [];
            document.getElementById("hotwords").value = list.join("\\n");
            setStatus("已加载", "ok");
        } catch (e) {
            console.error(e);
            setStatus("加载失败：" + e.message, "error");
        }
    }

    async function saveHotwords() {
        const text = document.getElementById("hotwords").value || "";
        setStatus("保存中...", "");
        try {
            const res = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ hotwords: text })
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || ("HTTP " + res.status));
            }
            const data = await res.json();
            const list = Array.isArray(data.hotwords) ? data.hotwords : [];
            document.getElementById("hotwords").value = list.join("\\n");
            setStatus("保存成功", "ok");
        } catch (e) {
            console.error(e);
            setStatus("保存失败：" + e.message, "error");
        }
    }

    document.getElementById("btn-save").addEventListener("click", saveHotwords);
    document.getElementById("btn-reload").addEventListener("click", loadHotwords);

    // 页面加载完成自动拉一次
    window.addEventListener("load", loadHotwords);
</script>
</body>
</html>
"""
        return web.Response(
            text=html,
            content_type="text/html",
            charset="utf-8",
        )

