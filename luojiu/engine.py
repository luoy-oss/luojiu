from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import threading
import time
from contextlib import contextmanager

from .brain import brain_from_state
from .learning import lexical_similarity, normalize
from .storage import Store


class UserError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def clean(value, name: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise UserError(f"{name}不能为空，且不能超过 {maximum} 个字符")
    return value.strip()


def password_hash(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 180_000).hex()
    return f"{salt}${digest}"


class Engine:
    def __init__(self, store: Store):
        self.store = store
        self.lock = threading.RLock()

    @contextmanager
    def transaction(self):
        with self.lock, self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    def _event(self, db, group, kind, detail):
        db.execute("INSERT INTO events(group_id,kind,detail,created) VALUES(?,?,?,?)",
                   (group, kind, detail, time.time()))

    def _member(self, db, user, group, owner=False):
        row = db.execute("SELECT g.* FROM groups g JOIN members m ON g.id=m.group_id "
                         "WHERE g.id=? AND m.user_id=?", (group, user)).fetchone()
        if not row:
            raise UserError("无权访问这个群", 403)
        if owner and row["owner"] != user:
            raise UserError("只有群主可以确认公共知识或管理群设置", 403)
        return row

    def register(self, user_id, name, password):
        user_id = clean(user_id, "用户 ID", 40)
        name = clean(name, "昵称", 40)
        password = clean(password, "密码", 200)
        if not re.fullmatch(r"[a-zA-Z0-9_-]{2,40}", user_id) or user_id.lower() == "luojiu":
            raise UserError("用户 ID 使用 2–40 位英文、数字、下划线或短横线，不能使用 luojiu")
        if len(password) < 8:
            raise UserError("密码至少 8 位")
        hashed = password_hash(password)
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone():
                raise UserError("这个用户 ID 已注册", 409)
            db.execute("INSERT INTO users VALUES(?,?,?,?)", (user_id, name, hashed, time.time()))
            return self._session(db, user_id, name)

    def _session(self, db, user_id, name):
        token = secrets.token_urlsafe(32)
        now = time.time()
        db.execute("DELETE FROM sessions WHERE expires<?", (now,))
        db.execute("INSERT INTO sessions VALUES(?,?,?)",
                   (hashlib.sha256(token.encode()).hexdigest(), user_id, now + 7 * 86400))
        return {"token": token, "user": {"id": user_id, "name": name}}

    def login(self, user_id, password):
        user_id = clean(user_id, "用户 ID", 40)
        password = clean(password, "密码", 200)
        with self.transaction() as db:
            row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            # Do comparable work even for an unknown user.
            stored = row["password"] if row else password_hash("unavailable", "0" * 32)
            candidate = password_hash(password, stored.split("$")[0])
            if not row or not hmac.compare_digest(candidate, stored):
                raise UserError("用户 ID 或密码不正确", 401)
            return self._session(db, row["id"], row["name"])

    def authenticate(self, token):
        with self.store.connect() as db:
            row = db.execute("SELECT user_id FROM sessions WHERE token=? AND expires>?",
                             (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
        if not row:
            raise UserError("请先登录", 401)
        return row["user_id"]

    def logout(self, token):
        with self.transaction() as db:
            db.execute("DELETE FROM sessions WHERE token=?", (hashlib.sha256(token.encode()).hexdigest(),))
        return {"ok": True}

    def groups(self, user):
        with self.store.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT g.id,g.name,g.owner FROM groups g JOIN members m ON g.id=m.group_id "
                "WHERE m.user_id=? ORDER BY g.created", (user,))]

    def create_group(self, user, name):
        name = clean(name, "群名称", 60)
        with self.transaction() as db:
            if db.execute("SELECT count(*) FROM groups WHERE owner=?", (user,)).fetchone()[0] >= 20:
                raise UserError("每个用户最多创建 20 个群")
            group, invite = secrets.token_hex(8), secrets.token_urlsafe(12)
            db.execute("INSERT INTO groups VALUES(?,?,?,?,?)", (group, name, invite, user, time.time()))
            db.execute("INSERT INTO members VALUES(?,?)", (group, user))
            db.execute("INSERT INTO bot_state(group_id,updated) VALUES(?,?)", (group, time.time()))
            self._event(db, group, "birth", "洛玖加入了群聊。尚无对话样本，也没有已形成的表达偏好。")
            return {"id": group, "name": name, "owner": user, "invite": invite}

    def join_group(self, user, invite):
        invite = clean(invite, "邀请码", 100)
        with self.transaction() as db:
            row = db.execute("SELECT id,name,owner FROM groups WHERE invite=?", (invite,)).fetchone()
            if not row:
                raise UserError("邀请码不正确", 404)
            db.execute("INSERT OR IGNORE INTO members VALUES(?,?)", (row["id"], user))
            return dict(row)

    def create_dm(self, user, target):
        target = clean(target, "对方用户 ID", 40)
        with self.transaction() as db:
            target_row = db.execute("SELECT id,name FROM users WHERE id=?", (target,)).fetchone()
            me = db.execute("SELECT name FROM users WHERE id=?", (user,)).fetchone()
            if not target_row:
                raise UserError("对方用户不存在", 404)
            if target == user:
                raise UserError("不能和自己创建私聊")
            existing = db.execute("SELECT g.id,g.name,g.owner FROM groups g JOIN members m ON g.id=m.group_id "
                                  "WHERE g.owner=? AND g.name=?", (user, "dm:" + target)).fetchone()
            if existing:
                return {"id": existing["id"], "name": existing["name"], "owner": existing["owner"], "private": True}
            group, invite = secrets.token_hex(8), secrets.token_urlsafe(12)
            db.execute("INSERT INTO groups VALUES(?,?,?,?,?)", (group, "dm:" + target, invite, user, time.time()))
            db.executemany("INSERT INTO members VALUES(?,?)", [(group, user), (group, target)])
            db.execute("INSERT INTO bot_state(group_id,updated) VALUES(?,?)", (group, time.time()))
            self._event(db, group, "birth", "private conversation created")
            return {"id": group, "name": me["name"] + " / " + target_row["name"], "owner": user, "private": True}

    def rotate_invite(self, user, group):
        with self.transaction() as db:
            self._member(db, user, group, True)
            invite = secrets.token_urlsafe(12)
            db.execute("UPDATE groups SET invite=? WHERE id=?", (invite, group))
            return {"invite": invite}

    def _model(self, db, group):
        row = db.execute("SELECT state FROM models WHERE group_id=?", (group,)).fetchone()
        return brain_from_state(json.loads(row["state"]) if row else None)

    def _save_model(self, db, group, model):
        db.execute("INSERT INTO models VALUES(?,?) ON CONFLICT(group_id) DO UPDATE SET state=excluded.state",
                   (group, json.dumps(model.state(), separators=(",", ":"))))

    def _state(self, db, group):
        row = db.execute("SELECT * FROM bot_state WHERE group_id=?", (group,)).fetchone()
        if row:
            return row
        db.execute("INSERT INTO bot_state(group_id,updated) VALUES(?,?)", (group, time.time()))
        return db.execute("SELECT * FROM bot_state WHERE group_id=?", (group,)).fetchone()

    def _feel(self, db, group, text, spoke=False):
        """Update a tiny persistent affect state from events, bounded to avoid drift."""
        state = self._state(db, group)
        now = time.time()
        lower = text.lower()
        # Affect is inferred from language-neutral surface signals.  Semantic
        # interpretation belongs to the brain and can be replaced later.
        warmth = 0.04 if re.search(r"[!！~😊🙂😂❤♥]", text) else 0
        stress = 0.05 if re.search(r"[!?！？💢😢😭]", text) and len(text) < 16 else 0
        novelty = min(0.12, max(0.0, len(normalize(text)) / 1600))
        energy = min(1.0, max(0.05, state["energy"] + warmth - stress - 0.006))
        curiosity = min(1.0, max(0.05, state["curiosity"] + novelty + (0.015 if "?" in text or "？" in text else 0) - 0.003))
        loneliness = max(0.0, state["loneliness"] - 0.08) if text else min(1.0, state["loneliness"] + 0.02)
        mood = "兴奋" if energy > 0.78 and curiosity > 0.65 else "有点担心" if stress > warmth and stress else "好奇" if curiosity > 0.68 else "平静"
        db.execute("UPDATE bot_state SET mood=?,energy=?,curiosity=?,loneliness=?,last_seen=?,last_spoke=CASE WHEN ? THEN ? ELSE last_spoke END,turn_count=turn_count+1,updated=? WHERE group_id=?",
                   (mood, energy, curiosity, loneliness, now, int(spoke), now if spoke else state["last_spoke"], now, group))

    def _conversation_signal(self, text, explicit=False):
        value = normalize(text)
        score = 0.0
        if explicit:
            score += 0.72
        if "?" in text or "？" in text:
            score += 0.42
        if any(mark in text for mark in ("!", "！", "…", "~")):
            score += 0.12
        if len(value) >= 18:
            score += 0.12
        return min(1.0, score)

    def _social_answer(self, db, group, user, text):
        """Return only language learned from examples; no phrase dictionary."""
        return None

    def _negatives(self, db, group, question, label):
        rows = db.execute("SELECT answer_id,question FROM examples WHERE group_id=? AND status='active' "
                          "AND answer_id!=?", (group, label)).fetchall()
        rows = sorted(rows, key=lambda r: lexical_similarity(question, r["question"]), reverse=True)
        return list(dict.fromkeys(str(r["answer_id"]) for r in rows))[:12]

    def _teach(self, db, user, group, question, answer, source, approve):
        question, answer = clean(question, "问题", 300), clean(answer, "回答", 1000)
        if not normalize(question):
            raise UserError("问题需要包含实际文字")
        existing = db.execute("SELECT e.* FROM examples e JOIN answers a ON a.id=e.answer_id "
                              "WHERE e.group_id=? AND e.question=? AND a.text=?",
                              (group, question, answer)).fetchone()
        if existing:
            if approve and existing["status"] != "active":
                self._activate(db, group, existing["id"])
            return {"id": existing["id"], "status": "active" if approve else existing["status"], "duplicate": True}
        if db.execute("SELECT count(*) FROM examples WHERE group_id=?", (group,)).fetchone()[0] >= 5000:
            raise UserError("本群已达到 5000 条样本上限，请导出整理后再学习")
        now = time.time()
        db.execute("INSERT OR IGNORE INTO answers(group_id,text,created) VALUES(?,?,?)", (group, answer, now))
        label = db.execute("SELECT id FROM answers WHERE group_id=? AND text=?", (group, answer)).fetchone()[0]
        cursor = db.execute("INSERT INTO examples(group_id,question,answer_id,author,status,source,created) "
                            "VALUES(?,?,?,?,?,?,?)", (group, question, label, user, "pending", source, now))
        example_id = cursor.lastrowid
        if approve:
            self._activate(db, group, example_id)
        else:
            self._event(db, group, "candidate", f"观察到一个候选回答，等待群主确认：{question[:70]}")
        return {"id": example_id, "status": "active" if approve else "pending", "duplicate": False}

    def _activate(self, db, group, example_id):
        row = db.execute("SELECT * FROM examples WHERE id=? AND group_id=?", (example_id, group)).fetchone()
        if not row:
            raise UserError("样本不存在", 404)
        if row["status"] == "active":
            return
        # A newly confirmed correction replaces contradictory examples for the same normalized question.
        other = db.execute("SELECT id,question,answer_id FROM examples WHERE group_id=? AND status='active' "
                           "AND answer_id!=?", (group, row["answer_id"])).fetchall()
        model = self._model(db, group)
        for previous in other:
            if normalize(previous["question"]) == normalize(row["question"]):
                db.execute("UPDATE examples SET status='superseded' WHERE id=?", (previous["id"],))
                db.execute("INSERT OR IGNORE INTO counterexamples VALUES(?,?,?)",
                           (group, normalize(row["question"]), previous["answer_id"]))
        db.execute("UPDATE examples SET status='active' WHERE id=?", (example_id,))
        db.execute("DELETE FROM counterexamples WHERE group_id=? AND question=? AND answer_id=?",
                   (group, normalize(row["question"]), row["answer_id"]))
        examples = [dict(r) for r in db.execute("SELECT e.*,a.text answer FROM examples e JOIN answers a ON e.answer_id=a.id WHERE e.group_id=? AND e.status='active'", (group,))]
        answer = db.execute("SELECT text FROM answers WHERE id=?", (row["answer_id"],)).fetchone()[0]
        model.teach(row["question"], answer, str(row["answer_id"]), examples)
        self._save_model(db, group, model)
        self._event(db, group, "learn", f"学会了一个经确认的回答，并更新了模型参数：{row['question'][:70]}")

    def teach(self, user, group, question, answer):
        with self.transaction() as db:
            g = self._member(db, user, group)
            return self._teach(db, user, group, question, answer, "teaching", g["owner"] == user)

    def review(self, user, group, example_id, approve):
        with self.transaction() as db:
            self._member(db, user, group, True)
            row = db.execute("SELECT status FROM examples WHERE id=? AND group_id=?", (example_id, group)).fetchone()
            if not row:
                raise UserError("样本不存在", 404)
            if approve:
                self._activate(db, group, example_id)
            else:
                db.execute("UPDATE examples SET status='rejected' WHERE id=?", (example_id,))
                self._event(db, group, "forget", "一条回答样本被停用，不再作为回答来源。")
                self._rebuild(db, group)
        return {"ok": True}

    def _rebuild(self, db, group):
        model = brain_from_state(None)
        rows = db.execute("SELECT question,answer_id FROM examples WHERE group_id=? AND status='active' ORDER BY id",
                          (group,)).fetchall()
        for row in rows:
            # Rebuild with bounded adjacent negatives; avoid quadratic corpus scans.
            answer = db.execute("SELECT text FROM answers WHERE id=?", (row["answer_id"],)).fetchone()[0]
            examples = [dict(r) for r in rows]
            model.teach(row["question"], answer, str(row["answer_id"]), examples)
        self._save_model(db, group, model)

    def _remember(self, db, group, user, text, message_id):
        # Memory extraction is intentionally language-neutral.  The shell has
        # one explicit command so any future language can teach the same slot:
        # /remember category=value.  Natural-language understanding belongs to
        # the replaceable brain and is never hard-coded here.
        match = re.fullmatch(r"\s*/remember\s+([A-Za-z0-9_.-]{1,40})=(.{1,160})\s*", text, re.S)
        if not match:
            return []
        patterns = [(match.group(1), None)]
        learned = []
        for category, _ in patterns:
            value = match.group(2).strip()
            exists = db.execute("SELECT 1 FROM facts WHERE group_id=? AND user_id=? AND category=? AND value=? "
                                "AND active=1", (group, user, category, value)).fetchone()
            if exists:
                continue
            db.execute("INSERT INTO facts(group_id,user_id,category,value,source_id,created) VALUES(?,?,?,?,?,?)",
                       (group, user, category, value, message_id, time.time()))
            learned.append((category, value))
        if learned:
            self._event(db, group, "memory", f"根据用户 {user} 本人的陈述更新了个人记忆。")
        return learned

    def _personal_answer(self, db, group, user, text):
        match = re.fullmatch(r"\s*/recall\s+([A-Za-z0-9_.-]{1,40})\s*", text, re.S)
        if not match:
            return None
        category = match.group(1)
        rows = db.execute("SELECT value,source_id FROM facts WHERE group_id=? AND user_id=? "
                          "AND category=? AND active=1 ORDER BY id DESC LIMIT 12", (group, user, category)).fetchall()
        if not rows:
            return ("我还没有记住这件事，你可以亲口告诉我。", 0, "还没有这个用户的相关记忆")
        prefix = f"{category}: "
        return (prefix + ", ".join(r["value"] for r in rows), 1.0,
                "来自本人陈述，消息 " + ", ".join(str(r["source_id"]) for r in rows))

    def _predict(self, db, group, question):
        examples = db.execute("SELECT e.id,e.question,e.answer_id,a.text answer FROM examples e "
                              "JOIN answers a ON e.answer_id=a.id WHERE e.group_id=? AND e.status='active'", (group,)).fetchall()
        if not examples:
            return None
        brain = self._model(db, group)
        result = brain.respond(question, examples)
        self._save_model(db, group, brain)
        if not result:
            return None
        row = next((r for r in examples if r["id"] == result.example_id), None)
        if not row:
            return None
        similarity = lexical_similarity(question, row["question"])
        return result.confidence, similarity, row

    def _style(self, db, group, answer):
        traits = {r["trait"]: r["evidence"] for r in db.execute("SELECT * FROM traits WHERE group_id=?", (group,))}
        if traits.get("concise", 0) >= 3:
            # Only shorten multi-sentence answers; never invent extra content.
            chunks = re.split(r"(?<=[。！？!?])", answer)
            if len(chunks) > 2 and len(chunks[0]) >= 8:
                return chunks[0]
        return answer

    def send(self, user, group, text, reply_to=None):
        text = clean(text, "消息", 2000)
        with self.transaction() as db:
            g = self._member(db, user, group)
            person = db.execute("SELECT name FROM users WHERE id=?", (user,)).fetchone()
            parent = None
            if reply_to is not None:
                parent = db.execute("SELECT * FROM messages WHERE id=? AND group_id=?", (reply_to, group)).fetchone()
                if not parent:
                    raise UserError("所回复的消息不在本群", 400)
            cursor = db.execute("INSERT INTO messages(group_id,user_id,name,text,created,reply_to) VALUES(?,?,?,?,?,?)",
                                (group, user, person["name"], text, time.time(), reply_to))
            message_id = cursor.lastrowid
            self._feel(db, group, text)
            live_brain = self._model(db, group)
            live_brain.perceive({"user_id": user, "text": text, "message_id": message_id, "time": time.time()})
            self._save_model(db, group, live_brain)
            remembered = self._remember(db, group, user, text, message_id)
            # Human-to-human explicit replies are observed as unconfirmed training candidates.
            if parent and parent["kind"] == "human" and parent["user_id"] != user:
                is_question = "?" in parent["text"] or "？" in parent["text"]
                if is_question and "?" not in text and "？" not in text and len(parent["text"]) <= 300 and len(text) <= 1000:
                    self._teach(db, user, group, parent["text"], text, "observed_reply", False)
            # A social agent decides whether to take a turn. This is deliberately
            # conservative: being silent is a valid action in a group.
            explicit = "@" in text or (parent and parent["kind"] != "human")
            question = "?" in text or "？" in text
            social = self._social_answer(db, group, user, text)
            state = self._state(db, group)
            signal = self._conversation_signal(text, explicit)
            recent_bot = db.execute("SELECT 1 FROM messages WHERE group_id=? AND kind!='human' AND created>? LIMIT 1",
                                    (group, time.time() - 90)).fetchone()
            # A stable hash makes the occasional unprompted turn reproducible and
            # avoids requiring a random service; curiosity and loneliness affect it.
            impulse = (int(hashlib.blake2s(f"{group}:{message_id}".encode(), digest_size=2).hexdigest(), 16) % 1000) / 1000
            spontaneous = (not parent and not recent_bot and impulse < (0.08 + 0.18 * state["curiosity"] + 0.10 * state["loneliness"]))
            should_reply = bool(explicit or question or remembered or social or spontaneous or signal >= 0.55)
            if not should_reply:
                return {"message_id": message_id, "reply_id": None}
            answer, label, confidence, reason, kind = "", None, 0.0, "", "unknown"
            if remembered:
                answer = "我记住了。这条记忆属于你的用户 ID；以后有变化也可以告诉我。"
                confidence, reason, kind = 1.0, "直接记录本人陈述", "memory"
            else:
                personal = self._personal_answer(db, group, user, text)
                prediction = None if personal or social else self._predict(db, group, text)
                if social:
                    answer, confidence, reason, kind = social, 0.72, "基于当前消息的情绪和社交信号", "social"
                elif personal:
                    answer, confidence, reason = personal
                    kind = "memory" if confidence else "unknown"
                elif prediction:
                    score, similarity, row = prediction
                    answer = self._style(db, group, row["answer"])
                    label, confidence, kind = row["answer_id"], score, "learned"
                    reason = f"样本 #{row['id']}；文字相似度 {similarity:.2f}；在线模型与检索综合分数 {score:.2f}（不是正确率）"
                else:
                    if spontaneous:
                        state = self._state(db, group)
                        answer = "刚才看到你们聊到这个，我有点好奇。你们怎么看？"
                        reason = "基于好奇心和群聊节奏主动加入"
                        kind = "social"
                    else:
                        answer = "这个我还不会。你们可以继续聊，我会先听着；如果有人愿意教我，我会把示范记下来。"
                        reason = "没有足够相近且经确认的知识，或候选答案存在歧义"
            self._feel(db, group, text, spoke=True)
            cursor = db.execute("INSERT INTO messages(group_id,user_id,name,text,created,reply_to,kind,label_id,confidence,reason,query) "
                                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                (group, "luojiu", "洛玖", answer, time.time(), message_id, kind, label, confidence, reason, text))
            return {"message_id": message_id, "reply_id": cursor.lastrowid}

    def autonomous_tick(self, group):
        """Let time pass through the agent even when no HTTP request arrives.

        A tick may produce one grounded social turn, or may only update affect
        and rehearsal. It never fabricates facts and never trains on its own text.
        """
        with self.transaction() as db:
            g = db.execute("SELECT id FROM groups WHERE id=?", (group,)).fetchone()
            if not g:
                return None
            state = self._state(db, group)
            now = time.time()
            last_human = db.execute("SELECT * FROM messages WHERE group_id=? AND kind='human' ORDER BY id DESC LIMIT 1", (group,)).fetchone()
            if not last_human:
                return None
            if now - last_human["created"] < 120 or now - state["last_spoke"] < 300:
                return None
            members = db.execute("SELECT count(*) FROM members WHERE group_id=?", (group,)).fetchone()[0]
            if members < 1 or state["loneliness"] < 0.35:
                # Quiet reflection: loneliness grows slowly while she waits.
                db.execute("UPDATE bot_state SET loneliness=min(1.0,loneliness+0.015),energy=max(0.05,energy-0.003),updated=? WHERE group_id=?", (now, group))
                return None
            # She only starts a topic occasionally, in a small group or when curious.
            impulse = int(hashlib.blake2s(f"tick:{group}:{int(now // 300)}".encode(), digest_size=2).hexdigest(), 16) % 100
            if impulse > 14 + int(state["curiosity"] * 20):
                db.execute("UPDATE bot_state SET loneliness=min(1.0,loneliness+0.02),updated=? WHERE group_id=?", (now, group))
                return None
            answer = "刚才的聊天让我想到一个问题：如果要把今天留下一句话，你们会写什么？"
            cursor = db.execute("INSERT INTO messages(group_id,user_id,name,text,created,kind,confidence,reason) VALUES(?,?,?,?,?,?,?,?)",
                                (group, "luojiu", "洛玖", answer, now, "social", 0.62, "经过等待、好奇心和群聊活跃度判断后主动发起话题"))
            self._feel(db, group, answer, spoke=True)
            self._event(db, group, "reflection", "洛玖等待了一段时间后，主动发起了一个开放话题。")
            return {"message_id": cursor.lastrowid, "text": answer}

    def feedback(self, user, group, message_id, value, correction=None, style=None):
        if value not in (-1, 1) or isinstance(value, bool):
            raise UserError("反馈只能是 1 或 -1")
        if style not in (None, "concise", "detailed"):
            raise UserError("不支持的表达偏好")
        if correction is not None:
            correction = clean(correction, "纠正内容", 1000)
        with self.transaction() as db:
            g = self._member(db, user, group)
            row = db.execute("SELECT * FROM messages WHERE id=? AND group_id=? AND kind!='human'",
                             (message_id, group)).fetchone()
            if not row:
                raise UserError("只能评价本群洛玖的回答", 404)
            if db.execute("SELECT 1 FROM feedback WHERE message_id=? AND user_id=?", (message_id, user)).fetchone():
                raise UserError("你已经评价过这条回答", 409)
            db.execute("INSERT INTO feedback VALUES(?,?,?,?,?)", (message_id, user, value, style, time.time()))
            # Only the owner can directly change shared model weights; other users' corrections are reviewed.
            if row["label_id"] and g["owner"] == user:
                model = self._model(db, group)
                model.feedback(row["query"], str(row["label_id"]), value)
                self._save_model(db, group, model)
                if value == -1:
                    db.execute("INSERT OR IGNORE INTO counterexamples VALUES(?,?,?)",
                               (group, normalize(row["query"]), row["label_id"]))
                else:
                    db.execute("DELETE FROM counterexamples WHERE group_id=? AND question=? AND answer_id=?",
                               (group, normalize(row["query"]), row["label_id"]))
            if correction:
                self._teach(db, user, group, row["query"][:300], correction, "correction", g["owner"] == user)
            if style and value == 1 and row["kind"] == "learned":
                db.execute("INSERT INTO traits VALUES(?,?,1) ON CONFLICT(group_id,trait) DO UPDATE SET evidence=evidence+1",
                           (group, style))
                opposite = "detailed" if style == "concise" else "concise"
                db.execute("UPDATE traits SET evidence=max(0,evidence-1) WHERE group_id=? AND trait=?", (group, opposite))
            self._event(db, group, "feedback", f"收到用户 {user} 的{'肯定' if value == 1 else '纠正'}反馈。")
            return {"ok": True, "applied": g["owner"] == user}

    def messages(self, user, group, after=0, before=None):
        with self.store.connect() as db:
            self._member(db, user, group)
            if before is not None:
                rows = list(reversed(db.execute("SELECT * FROM messages WHERE group_id=? AND id<? ORDER BY id DESC LIMIT 100",
                                                (group, before)).fetchall()))
            elif after:
                rows = db.execute("SELECT * FROM messages WHERE group_id=? AND id>? ORDER BY id LIMIT 100", (group, after)).fetchall()
            else:
                rows = list(reversed(db.execute("SELECT * FROM messages WHERE group_id=? ORDER BY id DESC LIMIT 100", (group,)).fetchall()))
            return [dict(r) for r in rows]

    def snapshot(self, user, group):
        with self.store.connect() as db:
            g = self._member(db, user, group)
            counts = {r["status"]: r["n"] for r in db.execute("SELECT status,count(*) n FROM examples WHERE group_id=? GROUP BY status", (group,))}
            model = self._model(db, group)
            traits = [dict(r) for r in db.execute("SELECT trait,evidence FROM traits WHERE group_id=?", (group,))]
            members = [dict(r) for r in db.execute("SELECT u.id,u.name FROM users u JOIN members m ON m.user_id=u.id WHERE m.group_id=?", (group,))]
            facts = [dict(r) for r in db.execute("SELECT f.*,u.name FROM facts f JOIN users u ON f.user_id=u.id WHERE group_id=? AND active=1 ORDER BY f.id DESC LIMIT 200", (group,))]
            examples = [dict(r) for r in db.execute("SELECT e.*,a.text answer FROM examples e JOIN answers a ON e.answer_id=a.id "
                                                   "WHERE e.group_id=? ORDER BY CASE e.status WHEN 'pending' THEN 0 ELSE 1 END,e.id DESC LIMIT 200", (group,))]
            events = [dict(r) for r in db.execute("SELECT * FROM events WHERE group_id=? ORDER BY id DESC LIMIT 40", (group,))]
            n = counts.get("active", 0)
            brain_state = model.state()
            return {"group": {"id": group, "name": g["name"], "owner": g["owner"], "invite": g["invite"] if g["owner"] == user else None},
                    "counts": counts, "updates": brain_state.get("updates", 0), "parameter_count": len(brain_state.get("nodes", [])),
                    "brain": {"type": brain_state.get("type"), "nodes": len(brain_state.get("nodes", [])),
                              "events": brain_state.get("event_count", 0), "functional_links": len(brain_state.get("functional", {}))},
                    "stage": "初见" if n == 0 else "积累" if n < 30 else "练习" if n < 150 else "持续学习",
                    "traits": traits, "members": members, "facts": facts, "examples": examples, "events": events}

    def forget_fact(self, user, group, fact_id):
        with self.transaction() as db:
            self._member(db, user, group)
            row = db.execute("SELECT user_id FROM facts WHERE id=? AND group_id=?", (fact_id, group)).fetchone()
            if not row or row["user_id"] != user:
                raise UserError("只能移除关于你自己的记忆", 403)
            db.execute("UPDATE facts SET active=0 WHERE id=?", (fact_id,))
            self._event(db, group, "forget", f"按用户 {user} 的要求移除了一条个人记忆。")
        return {"ok": True}

    def consolidate(self):
        """Bounded spaced rehearsal, never training on the bot's own generated answers."""
        with self.transaction() as db:
            now = time.time()
            rows = db.execute("SELECT * FROM examples WHERE status='active' AND rehearsed<? ORDER BY rehearsed,id LIMIT 24",
                              (now - 3600,)).fetchall()
            grouped = {}
            for row in rows:
                grouped.setdefault(row["group_id"], []).append(row)
            for group, examples in grouped.items():
                model = self._model(db, group)
                learned = 0
                for row in examples:
                    blocked = db.execute("SELECT 1 FROM counterexamples WHERE group_id=? AND question=? AND answer_id=?",
                                         (group, normalize(row["question"]), row["answer_id"])).fetchone()
                    if not blocked:
                        answer = db.execute("SELECT text FROM answers WHERE id=?", (row["answer_id"],)).fetchone()[0]
                        active = [dict(r) for r in db.execute("SELECT e.*,a.text answer FROM examples e JOIN answers a ON e.answer_id=a.id WHERE e.group_id=? AND e.status='active'", (group,))]
                        model.teach(row["question"], answer, str(row["answer_id"]), active)
                        learned += 1
                    db.execute("UPDATE examples SET rehearsed=? WHERE id=?", (now, row["id"]))
                if learned:
                    self._save_model(db, group, model)
                    self._event(db, group, "rehearsal", f"复习了 {learned} 条已确认样本。复习不会增加新知识。")
            return len(rows)

    def export(self, user, group):
        with self.store.connect() as db:
            self._member(db, user, group, True)
            return {"format": "luojiu-examples-v1", "examples": [dict(r) for r in db.execute(
                "SELECT e.question,a.text answer FROM examples e JOIN answers a ON e.answer_id=a.id "
                "WHERE e.group_id=? AND e.status='active' ORDER BY e.id", (group,))]}

    def import_examples(self, user, group, examples):
        if not isinstance(examples, list) or not 1 <= len(examples) <= 200:
            raise UserError("每次导入 1–200 条问答")
        with self.transaction() as db:
            self._member(db, user, group, True)
            result = []
            for row in examples:
                if not isinstance(row, dict):
                    raise UserError("每条样本需要 question 和 answer 字段")
                result.append(self._teach(db, user, group, row.get("question"), row.get("answer"), "import", True))
            return {"count": len(result)}
