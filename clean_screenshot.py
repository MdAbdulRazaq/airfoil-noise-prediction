"""Capture all Phase 12 TASK 3 evidence screenshots in one clean pass.

Fresh Chrome tab -> navigate to the running Streamlit app -> wait for hydration
-> capture (1) initial page -> (2) click Predict, capture (3) prediction.
The setup-error screenshot (missing model -> clear setup message, no stack
trace) was already captured previously with the artifact absent and is valid.

No application code is modified; the model artifact is left untouched.
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
from pathlib import Path

import websocket

REPO = Path(__file__).resolve().parent
EVIDENCE = REPO / "reports" / "evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)
STREAMLIT_URL = "http://127.0.0.1:8501"
CDP_HTTP = "http://127.0.0.1:9223"


class CdpClient:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self._nid = 1

    def send(self, method: str, params=None):
        params = params or {}
        self._nid += 1
        self.ws.send(json.dumps({"id": self._nid, "method": method, "params": params}))
        deadline = time.time() + 40
        import json as j
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"timed out waiting for {method}")
            self.ws.settimeout(max(0.1, deadline - time.time()))
            raw = self.ws.recv()
            if raw is None:
                raise ConnectionError("websocket closed")
            msg = j.loads(raw)
            if msg.get("id") == self._nid:
                if "error" in msg:
                    raise RuntimeError(f"CDP error {method}: {msg['error']}")
                return msg.get("result", {})


def find_page_target():
    targets = json.loads(subprocess.check_output(
        ["curl", "-s", f"{CDP_HTTP}/json"], text=True))
    pages = [t for t in targets if t.get("type") == "page"]
    if not pages:
        raise RuntimeError("no Chrome page targets")
    return pages[0]


def wait_until(client, js, timeout=45, poll=0.4):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        res = client.send("Runtime.evaluate", {
            "expression": js, "returnByValue": True})
        last = res.get("result", {}).get("value")
        if last:
            return last
        time.sleep(poll)
    return last


def screenshot(client, name: str, label: str):
    result = client.send("Page.captureScreenshot", {"format": "png"})
    data = base64.b64decode(result["data"])
    path = EVIDENCE / name
    path.write_bytes(data)
    print(f"[screenshot] {label} -> {path} ({len(data)} bytes)")
    return path


def click_predict(client):
    client.send("Runtime.evaluate", {
        "expression": """
        (() => {
          const bs = Array.from(document.querySelectorAll('button'));
          const b = bs.find(e => (e.innerText || '').trim() === 'Predict');
          if (!b) return false;
          b.click();
          return true;
        })()
        """,
        "returnByValue": True,
    })


def main():
    target = find_page_target()
    print("target:", target.get("title"), target.get("url"))
    client = CdpClient(target["webSocketDebuggerUrl"])
    for domain in ("Page", "DOM", "Runtime"):
        client.send(f"{domain}.enable")

    client.send("Page.navigate", {"url": STREAMLIT_URL})
    wait_until(client, 'document.body.innerText.includes("Frequency (Hz)"'
               ' && document.body.innerText.includes("Predict")', timeout=60)
    time.sleep(1.0)
    print("--- initial body ---")
    print(client.send("Runtime.evaluate", {
        "expression": "document.body.innerText || ''",
        "returnByValue": True})["result"]["value"][:300])
    screenshot(client, "screenshot_initial.png", "Initial frontend page")

    click_predict(client)
    ok = wait_until(client, 'document.body.innerText.includes("126.40 dB")', timeout=40)
    print(f"[check] prediction '126.40 dB' appeared: {ok}")
    print("--- post-predict body ---")
    print(client.send("Runtime.evaluate", {
        "expression": "document.body.innerText || ''",
        "returnByValue": True})["result"]["value"][:700])
    screenshot(client, "screenshot_prediction.png", "Successful prediction result (~126.40 dB)")
    print("\ndone")


if __name__ == "__main__":
    main()
