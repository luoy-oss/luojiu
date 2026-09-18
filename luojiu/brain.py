"""Pluggable brains for Luo Jiu.

The shell never imports an implementation detail from this module.  A Brain
receives normalized observations and delayed teaching signals.  SMGrNN is a
small self-motivated growing neural network: hashed sparse vectors are routed
to local nodes; teaching grows, moves, merges, and decays nodes.  There is no
back-propagation and no dependency on a tokenizer or an embedding service.
"""
from __future__ import annotations

import abc
import hashlib
import math
import random
import time
from dataclasses import dataclass

from .learning import features, lexical_similarity, normalize


@dataclass
class BrainReply:
    text: str
    confidence: float
    reason: str
    label: str | None = None
    example_id: int | None = None


class Brain(abc.ABC):
    """Stable contract between the durable shell and a replaceable brain."""

    @abc.abstractmethod
    def perceive(self, observation: dict) -> None: ...

    @abc.abstractmethod
    def respond(self, text: str, examples: list[dict]) -> BrainReply | None: ...

    @abc.abstractmethod
    def teach(self, question: str, answer: str, label: str, examples: list[dict]) -> None: ...

    @abc.abstractmethod
    def feedback(self, question: str, label: str, value: int) -> None: ...

    @abc.abstractmethod
    def state(self) -> dict: ...


class SMGrNN(Brain):
    """Self-Motivated Growing Neural Network using only local updates.

    Nodes hold sparse prototypes.  A teaching signal attracts the nearest
    node, creates a node for novel input, and merges close nodes.  Every load
    decays old node energy, allowing capacity to change over a long lifetime.
    """

    def __init__(self, state: dict | None = None):
        state = state or {}
        self.nodes = state.get("nodes", [])
        self.updates = int(state.get("updates", 0))
        self.observations = int(state.get("observations", 0))
        self.last_rehearsal = float(state.get("last_rehearsal", 0))
        self.rng = random.Random(int(state.get("seed", 90210)))
        self.event_count = int(state.get("event_count", 0))
        self.functional = state.get("functional", {})
        self._normalize_nodes()
        self._decay()

    def _normalize_nodes(self):
        """Add membrane and synapse state to nodes created by older versions."""
        for node in self.nodes:
            node.setdefault("potential", 0.0)
            node.setdefault("inhibition", 0.0)
            node.setdefault("excitation", 0.0)
            node.setdefault("rate", 0.0)
            node.setdefault("threshold", 0.72)
            node.setdefault("excitatory", True)
            node.setdefault("synapses", {})

    @staticmethod
    def _distance(left: dict[str, float], right: dict[str, float]) -> float:
        keys = set(left) | set(right)
        return math.sqrt(sum((left.get(k, 0.0) - right.get(k, 0.0)) ** 2 for k in keys))

    @staticmethod
    def _centroid(left: dict[str, float], right: dict[str, float], rate: float) -> dict[str, float]:
        keys = set(left) | set(right)
        return {k: left.get(k, 0.0) + rate * (right.get(k, 0.0) - left.get(k, 0.0))
                for k in keys if abs(left.get(k, 0.0) + rate * (right.get(k, 0.0) - left.get(k, 0.0))) > 0.004}

    def _decay(self):
        now = time.time()
        for node in self.nodes:
            age = max(0.0, now - float(node.get("touched", now)))
            node["energy"] = max(0.01, float(node.get("energy", 0.25)) * math.exp(-age / (30 * 86400)))
        if len(self.nodes) > 8:
            self.nodes = [n for n in self.nodes if n.get("energy", 0) > 0.02 or n.get("wins", 0) > 2]

    def perceive(self, observation: dict) -> None:
        # Perception is intentionally separate from learning.  It changes only
        # homeostatic counters; no observation can add a training label.
        self.observations += 1
        self._pulse_train(features(str(observation.get("text", ""))), plastic=False)

    def _pulse_train(self, vector: dict[str, float], plastic: bool):
        """Run a small asynchronous event stream.

        Inter-arrival times are sampled independently (Poisson-like), with no
        global tick and no matrix batch.  Excitation and inhibition are local
        state variables; the homeostatic clamp prevents either population from
        winning permanently.
        """
        if not self.nodes:
            return []
        active = sorted(((abs(v), k) for k, v in vector.items() if k != "bias"), reverse=True)[:24]
        if not active:
            return []
        spikes = []
        clock = 0.0
        for amplitude, feature in active:
            rate = 2.0 + 24.0 * min(1.0, amplitude)
            # Exponential waiting time creates irregular events.
            clock += self.rng.expovariate(rate)
            target = self.rng.randrange(len(self.nodes))
            spikes.append((clock, target, amplitude, feature))
        spikes.sort(key=lambda item: item[0])
        fired = []
        for _, target, amplitude, feature in spikes:
            node = self.nodes[target]
            dt = max(0.001, _)
            for item in self.nodes:
                item["potential"] *= math.exp(-dt / 0.035)
                item["excitation"] *= math.exp(-dt / 0.080)
                item["inhibition"] *= math.exp(-dt / 0.045)
            node["excitation"] += amplitude * (1.0 if node.get("excitatory", True) else 0.7)
            node["potential"] += node["excitation"] - node["inhibition"]
            if node["potential"] >= node.get("threshold", 0.72):
                node["potential"] = 0.08
                node["rate"] = min(1.0, node.get("rate", 0.0) * 0.92 + 0.18)
                node["wins"] = int(node.get("wins", 0)) + 1
                node["touched"] = time.time()
                fired.append(target)
                # Each spike excites local structural neighbors and recruits a
                # nearby inhibitory cell. Functional co-activity is separate.
                for neighbor_id, weight in list(node.get("synapses", {}).items()):
                    neighbor = next((n for n in self.nodes if n["id"] == neighbor_id), None)
                    if neighbor:
                        if neighbor.get("excitatory", True):
                            neighbor["potential"] += 0.04 * weight
                        else:
                            neighbor["inhibition"] += 0.06 * weight
        self.event_count += len(spikes)
        for node in self.nodes:
            node["rate"] *= 0.985
            # Local inhibitory feedback maintains an E/I ratio near 65/35.
            node["inhibition"] += max(0.0, node["excitation"] - 0.65) * 0.17
            node["threshold"] = min(1.35, max(0.32, node.get("threshold", 0.72) + (node.get("rate", 0.0) - 0.11) * 0.008))
        for left in fired:
            for right in fired:
                if left != right:
                    key = self.nodes[left]["id"] + ":" + self.nodes[right]["id"]
                    self.functional[key] = min(1.0, self.functional.get(key, 0.0) * 0.99 + 0.025)
        return fired

    def _winner(self, vector: dict[str, float], label: str | None = None):
        candidates = [n for n in self.nodes if label is None or n.get("label") == label]
        if not candidates:
            return None, float("inf")
        ranked = sorted(((self._distance(vector, n["prototype"]), n) for n in candidates), key=lambda x: x[0])
        return ranked[0][1], ranked[0][0]

    def respond(self, text: str, examples: list[dict]) -> BrainReply | None:
        vector = features(text)
        fired = self._pulse_train(vector, plastic=False)
        winner, distance = self._winner(vector)
        if winner is None or distance > 1.10:
            return None
        # The node is only a routing decision.  The exact language comes from
        # a stored human teaching example, never from the bot's own output.
        candidates = [row for row in examples if str(row["answer_id"]) == str(winner["label"])]
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda row: lexical_similarity(text, row["question"]), reverse=True)
        similarity = lexical_similarity(text, ranked[0]["question"])
        activity = max((self.nodes[index].get("rate", 0.0) for index in fired), default=0.0)
        confidence = max(0.0, min(1.0, 0.52 * math.exp(-distance / 1.3) + 0.30 * similarity + 0.18 * activity))
        if confidence < 0.53:
            return None
        winner["wins"] = int(winner.get("wins", 0)) + 1
        winner["touched"] = time.time()
        return BrainReply(ranked[0]["answer"], confidence,
                          f"local_node={winner['id']}; distance={distance:.3f}; similarity={similarity:.3f}",
                          str(winner["label"]), ranked[0]["id"])

    def teach(self, question: str, answer: str, label: str, examples: list[dict]) -> None:
        vector = features(question)
        fired = self._pulse_train(vector, plastic=True)
        same, distance = self._winner(vector, label)
        now = time.time()
        anchor = None
        if same is None or distance > 0.82:
            digest = hashlib.blake2s(f"{label}:{question}".encode(), digest_size=5).hexdigest()
            self.nodes.append({"id": digest, "label": str(label), "prototype": vector,
                               "energy": 0.45, "wins": 0, "touched": now,
                               "potential": 0.0, "inhibition": 0.0, "excitation": 0.0,
                               "rate": 0.0, "threshold": 0.72,
                               "excitatory": len(self.nodes) % 3 != 0, "synapses": {}})
            anchor = len(self.nodes) - 1
        else:
            same["prototype"] = self._centroid(same["prototype"], vector, 0.22)
            same["energy"] = min(1.0, float(same.get("energy", 0.3)) + 0.09)
            same["touched"] = now
            anchor = self.nodes.index(same)
        self._normalize_nodes()
        if anchor is not None and anchor not in fired:
            fired.append(anchor)
        self._wire_local(fired)
        self.updates += 1
        self._merge()

    def _wire_local(self, fired: list[int]):
        """Grow sparse anatomical links from the recent local event neighborhood."""
        if len(self.nodes) < 2:
            return
        for index in fired[:8]:
            if index >= len(self.nodes):
                continue
            source = self.nodes[index]
            ranked = sorted(((self._distance(source["prototype"], other["prototype"]), other)
                             for other in self.nodes if other is not source), key=lambda item: item[0])[:3]
            for distance, target in ranked:
                source.setdefault("synapses", {})[target["id"]] = min(1.0, 0.2 + math.exp(-distance))

    def feedback(self, question: str, label: str, value: int) -> None:
        node, _ = self._winner(features(question), label)
        if not node:
            return
        node["energy"] = min(1.0, node.get("energy", 0.2) + 0.07) if value > 0 else max(0.01, node.get("energy", 0.2) * 0.62)
        node["touched"] = time.time()
        self.updates += 1

    def _merge(self):
        if len(self.nodes) < 2:
            return
        survivors = []
        for node in sorted(self.nodes, key=lambda item: item.get("energy", 0), reverse=True):
            close = next((item for item in survivors if item.get("label") == node.get("label") and
                          self._distance(item["prototype"], node["prototype"]) < 0.18), None)
            if close:
                close["prototype"] = self._centroid(close["prototype"], node["prototype"], 0.35)
                close["energy"] = min(1.0, close.get("energy", 0) + node.get("energy", 0) * 0.2)
            else:
                survivors.append(node)
        self.nodes = survivors[:256]

    def state(self) -> dict:
        return {"type": "SMGrNN", "version": 1, "nodes": self.nodes,
                "updates": self.updates, "observations": self.observations,
                "last_rehearsal": self.last_rehearsal, "seed": 90210,
                "event_count": self.event_count, "functional": self.functional}


def brain_from_state(state: dict | None) -> Brain:
    # The shell changes this one factory line when a future brain is installed.
    return SMGrNN(state)
