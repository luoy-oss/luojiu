from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .engine import Engine, UserError


class App:
    def __init__(self, engine: Engine, web: Path):
        self.engine, self.web = engine, web
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._clock, daemon=True, name="luojiu-clock")

    def start(self):
        self.thread.start()

    def _clock(self):
        while not self.stop.wait(20):
            try:
                groups = self.engine.groups("__clock__") if False else None
                with self.engine.store.connect() as db:
                    ids = [r[0] for r in db.execute("SELECT id FROM groups")]
                for group in ids:
                    self.engine.autonomous_tick(group)
                self.engine.consolidate()
            except Exception as exc:  # the clock must never take down the web server
                print(f"洛玖后台时钟异常: {exc}")

    def close(self):
        self.stop.set()


def make_handler(app: App):
    engine = app.engine

    class Handler(BaseHTTPRequestHandler):
        server_version = "Luojiu/1.0"

        def log_message(self, fmt, *args):
            # Keep the terminal useful; messages themselves are available in the UI.
            if self.path.startswith("/api") and args and args[1] not in ("200", "204"):
                super().log_message(fmt, *args)

        def _json(self, status, payload):
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _body(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1_000_000:
                    raise UserError("请求过大")
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                raise UserError("请求不是有效的 JSON")

        def _token(self):
            value = self.headers.get("Authorization", "")
            return value[7:] if value.startswith("Bearer ") else ""

        def _user(self):
            token = self._token()
            if not token:
                raise UserError("请先登录", 401)
            return engine.authenticate(token)

        def _call(self, fn):
            try:
                self._json(200, fn())
            except UserError as exc:
                self._json(exc.status, {"error": str(exc)})
            except Exception as exc:
                print(f"API error: {exc}")
                self._json(500, {"error": "服务器遇到内部错误"})

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                return self._json(200, {"ok": True, "name": "洛玖"})
            if not parsed.path.startswith("/api/"):
                return self._file(parsed.path)
            query = parse_qs(parsed.query)
            def call():
                user = self._user()
                if parsed.path == "/api/groups":
                    return engine.groups(user)
                if parsed.path == "/api/messages":
                    group = query.get("group", [""])[0]
                    after = int(query.get("after", [0])[0])
                    before = query.get("before", [None])[0]
                    return engine.messages(user, group, after, int(before) if before else None)
                if parsed.path == "/api/snapshot":
                    return engine.snapshot(user, query.get("group", [""])[0])
                if parsed.path == "/api/export":
                    return engine.export(user, query.get("group", [""])[0])
                raise UserError("没有这个接口", 404)
            return self._call(call)

        def do_POST(self):
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/"):
                return self._json(404, {"error": "没有这个接口"})
            def call():
                body = self._body()
                path = parsed.path
                if path == "/api/register":
                    return engine.register(body.get("id"), body.get("name"), body.get("password"))
                if path == "/api/login":
                    return engine.login(body.get("id"), body.get("password"))
                if path == "/api/logout":
                    return engine.logout(self._token())
                user = self._user()
                group = body.get("group")
                if path == "/api/groups/create":
                    return engine.create_group(user, body.get("name"))
                if path == "/api/groups/join":
                    return engine.join_group(user, body.get("invite"))
                if path == "/api/dm":
                    return engine.create_dm(user, body.get("target"))
                if path == "/api/groups/invite":
                    return engine.rotate_invite(user, group)
                if path == "/api/send":
                    return engine.send(user, group, body.get("text"), body.get("reply_to"))
                if path == "/api/teach":
                    return engine.teach(user, group, body.get("question"), body.get("answer"))
                if path == "/api/review":
                    return engine.review(user, group, int(body.get("example_id")), bool(body.get("approve")))
                if path == "/api/feedback":
                    return engine.feedback(user, group, int(body.get("message_id")), int(body.get("value")), body.get("correction"), body.get("style"))
                if path == "/api/fact/forget":
                    return engine.forget_fact(user, group, int(body.get("fact_id")))
                if path == "/api/import":
                    return engine.import_examples(user, group, body.get("examples"))
                raise UserError("没有这个接口", 404)
            return self._call(call)

        def _file(self, path):
            relative = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (app.web / relative).resolve()
            if app.web.resolve() not in target.parents and target != app.web.resolve():
                return self._json(404, {"error": "找不到页面"})
            if not target.is_file():
                return self._json(404, {"error": "找不到页面"})
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def run(database: str = "data/luojiu.sqlite3", host: str = "127.0.0.1", port: int = 8765):
    from .storage import Store
    root = Path(__file__).parent
    app = App(Engine(Store(database)), root / "web")
    server = ThreadingHTTPServer((host, port), make_handler(app))
    app.start()
    print(f"洛玖正在运行：http://{host}:{port}")
    print(f"数据文件：{Path(database).resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="运行洛玖本地群聊环境")
    parser.add_argument("--database", default="data/luojiu.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run(args.database, args.host, args.port)
