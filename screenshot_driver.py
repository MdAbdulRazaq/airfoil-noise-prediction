"""Capture evidence screenshots of the running Streamlit app (Phase 12, TASK 3).

Drives Chrome headless (remote-debugging-port=9223) against the Streamlit
server on http://127.0.0.1:8501. Waits for full Streamlit hydration and for
the server round-trip after each button click before capturing.

Captures:
  1. screenshot_initial.png   - initial frontend page (no prediction)
  2. screenshot_prediction.png - after clicking Predict (~126.40 dB)
  3. screenshot_setup_error.png - missing model artifact -> clear setup msg

No application code is modified; the model artifact is temporarily renamed for
the setup-error capture and restored immediately afterwards.
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

ARTIFACT = REPO / "models" / "best_model.joblib"
ARTIFACT_BACKUP = REPO / "models" / ".best_model.joblib.screenshot_backup"


class CdpClient:
    def __init__(self, ws_url: str, timeout: float = 30):
        self.ws = websocket.create_connection(ws_url, timeout=timeout)
        self._next_id = 1

    def send(self, method: str, params=None):
        params = params or {}
        msg_id = self._next_id
        self._next_id += 1
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        deadline = time.time() + 30
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {method}")
            self.ws.settimeout(max(0.1, remaining))
            raw = self.ws.recv()
            if raw is None:
                raise ConnectionError("WebSocket closed")
            msg = json.loads(raw)
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"CDP error {method}: {msg['error']}")
                return msg.get("result", {})


def find_page_target():
    targets = json.loads(subprocess.check_output(
        ["curl", "-s", f"{CDP_HTTP}/json"], text=True))
    pages = [t for t in targets if t.get("type") == "page"]
    if not pages:
        raise RuntimeError("No Chrome page targets found")
    return pages[0]


def wait_until(client, js, timeout=45, poll=0.5):
    """Poll JS (via Runtime.evaluate) until it returns a truthy value."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.send("Runtime.evaluate", {
            "expression": js,
            "returnByValue": True,
        })
        val = res.get("result", {}).get("value")
        if val:
            return val
        time.sleep(poll)
    return None


def body_text(client):
    return client.send("Runtime.evaluate", {
        "expression": "document.body.innerText || ''",
        "returnByValue": True,
    })["result"]["value"]


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
          const buttons = Array.from(document.querySelectorAll('button'));
          const b = buttons.find(el => (el.innerText || '').trim() === 'Predict');
          if (!b) return false;
          b.click();
          return true;
        })()
        """,
        "returnByValue": True,
    })


def main():
    target = find_page_target()
    print("Target:", target.get("title"))
    client = CdpClient(target["webSocketDebuggerUrl"])
    for domain in ("Page", "DOM", "Runtime"):
        client.send(f"{domain}.enable")

    # ---- Navigate and wait for full Streamlit hydration ------------------
    client.send("Page.navigate", {"url": STREAMLIT_URL})
    wait_until(client, 'document.body.innerText.includes("Frequency (Hz)"'
               ' && document.body.innerText.includes("Predict")', timeout=60)
    time.sleep(1.0)  # let widget styles settle
    print("--- hydrated body text sample ---")
    print(body_text(client)[:400])
    screenshot(client, "screenshot_initial.png", "Initial frontend page")

    # ---- 3. Missing model artifact -> clear setup error (BEFORE any predict
    #      so the model is not yet cached in the server) -------------------
    ARTIFACT.rename(ARTIFACT_BACKUP)
    try:
        click_predict(client)
        ok = wait_until(client, 'document.body.innerText.includes("Run the training workflow")', timeout=30)
        print(f"[check] setup message appeared: {ok}")
        print("--- body text after predict with missing artifact ---")
        print(body_text(client)[:600])
        screenshot(client, "screenshot_setup_error.png", "Missing model artifact -> setup message")
    finally:
        ARTIFACT_BACKUP.rename(ARTIFACT)

    # ---- 2. Successful prediction (defaults == first dataset row) --------
    click_predict(client)
    ok = wait_until(client, 'document.body.innerText.includes("126.40 dB")', timeout=30)
    print(f"[check] prediction '126.40 dB' appeared: {ok}")
    print("--- body text after predict ---")
    print(body_text(client)[:600])
    screenshot(client, "screenshot_prediction.png", "Successful prediction result (~126.40 dB)")

    print("\nEvidence screenshots captured.")


if __name__ == "__main__":
    main()
