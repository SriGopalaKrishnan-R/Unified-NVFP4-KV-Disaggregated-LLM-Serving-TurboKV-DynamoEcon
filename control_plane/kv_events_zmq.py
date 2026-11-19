"""ZeroMQ KV-Event Pub/Sub Bus matching NVIDIA Dynamo 1.x `--kv-events-config` schema."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List

import zmq


class DynamoZMQKVEventBus:
    """Thread-safe ZeroMQ PUB/SUB bus: SUB socket lives exclusively on its receiver thread."""

    def __init__(self, topic: str = "dynamo.kv_events.v1", max_history: int = 256) -> None:
        self.topic = topic
        self._ctx = zmq.Context()
        self._pub_lock = threading.Lock()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self.port = self._pub.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = f"tcp://127.0.0.1:{self.port}"

        self._lock = threading.Lock()
        self._seq_id = 0
        self._history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._running = True
        self._sub_ready = threading.Event()
        self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx_thread.start()
        self._sub_ready.wait(timeout=1.0)
        time.sleep(0.04)

    def _recv_loop(self) -> None:
        sub = self._ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0)
        sub.connect(self.endpoint)
        sub.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        self._sub_ready.set()
        try:
            while self._running:
                socks = dict(poller.poll(timeout=25))
                if sub in socks:
                    while True:
                        try:
                            raw = sub.recv_string(flags=zmq.NOBLOCK)
                            _, payload_str = raw.split(" ", 1)
                            ev = json.loads(payload_str)
                            with self._lock:
                                self._history.append(ev)
                        except zmq.Again:
                            break
        finally:
            poller.unregister(sub)
            sub.close(0)

    def publish_event(
        self,
        event_type: str,
        worker_id: int,
        block_hashes: List[str],
        parent_hash: str = "0x0",
        token_block_size: int = 16,
        s_retain: float = 0.0,
        lora_id: int = 0,
    ) -> Dict[str, Any]:
        with self._lock:
            self._seq_id += 1
            eid = self._seq_id

        envelope = {
            "schema": "dynamo.kv_events_config.v1",
            "event_id": eid,
            "timestamp_ns": time.time_ns(),
            "worker_id": worker_id,
            "dp_rank": 0,
            "event": {
                "type": event_type,  # BlockStored | BlockRemoved | AllBlocksCleared
                "block_hashes": block_hashes,
                "parent_block_hash": parent_hash,
                "token_block_size": token_block_size,
                "lora_id": lora_id,
                "tokconomics_s_retain": round(s_retain, 4),
            },
        }
        wire = f"{self.topic} {json.dumps(envelope)}"
        with self._pub_lock:
            if not self._pub.closed:
                self._pub.send_string(wire)
        return envelope

    def recent_events(self, limit: int = 32) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._history)
        return items[-limit:]

    def close(self) -> None:
        self._running = False
        self._rx_thread.join(timeout=1.0)
        with self._pub_lock:
            if not self._pub.closed:
                self._pub.close(0)
        self._ctx.term()
