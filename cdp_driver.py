"""CDP-driven verification of the running Streamlit app (Phase 12 acceptance).

Drives Chrome headless (remote-debugging-port=9222) against the Streamlit
server on http://127.0.0.1:8501, performs the Section 11.2 acceptance checks
against the *running* application, and saves evidence screenshots.

No application code is modified.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from pathlib import Path

import websocket

REPO = Path(__file__).resolve().parent
EVIDENCE = REPO / "reports" / "evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)

STREAMLIT_URL = "http://127.0.0.1:8501"
CDP_HTTP = "http://127.0.0.1:9223"

# First real dataset row (valid inputs).
FIRST_ROW = {
    "frequency_hz": 800.0,
    "angle_of_attack_deg": 0.0,
    "chord_length_m": 0.3048,
    "free_stream_velocity_m_s": 71.3,
    "suction_side_displacement_thickness_m": 0.00266337,
}

ARTIFACT = REPO / "models" / "best_model.joblib"
ARTIFACT_BACKUP = REPO / "models" / ".best_model.joblib.phase12_backup"


class CdpClient:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self._next_id = 1
        self._pending = {}

    def send(self, method: str, params=None, wait_event=None, timeout=30):
        params = params or {}
        msg_id = self._next_id
        self._next_id += 1
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        deadline = time.time() + timeout
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
            # Handle async events.
            if wait_event and msg.get("method") == wait_event:
                return msg.get("params", {})


def find_page_target():
    targets = json.loads(subprocess.check_output(
        ["curl", "-s", f"{CDP_HTTP}/json"], text=True))
    pages = [t for t in targets if t.get("type") == "page"]
    if not pages:
        raise RuntimeError("No Chrome page targets found")
    return pages[0]


def wait_for(client, js, poll=0.5, timeout=30):
    """Evaluate JS repeatedly until it returns a truthy value."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        res = client.send("Runtime.evaluate", {
            "expression": js,
            "returnByValue": True,
            "awaitPromise": True,
        })
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


def main():
    target = find_page_target()
    print("Target:", target.get("title"), target.get("url"))
    client = CdpClient(target["webSocketDebuggerUrl"])
    for domain in ("Page", "DOM", "Runtime"):
        client.send(f"{domain}.enable")

    # ---- 1. Navigate to the running app ----------------------------------
    client.send("Page.navigate", {"url": STREAMLIT_URL})
    # Wait until Streamlit has rendered the Predict button.
    wait_for(client, '!!document.querySelector("button")', timeout=60)
    time.sleep(1.0)  # let Streamlit finish hydrating widgets

    screenshot(client, "screenshot_initial.png", "Initial frontend page")

    # Count buttons (expect exactly one: "Predict").
    n_buttons = client.send("Runtime.evaluate", {
        "expression": "document.querySelectorAll('button').length",
        "returnByValue": True,
    })["result"]["value"]
    print(f"[check] buttons on page: {n_buttons}")

    # ---- 2. Missing model artifact -> clear setup message ----------------
    # No prediction has been made yet, so the model is not cached. Rename the
    # artifact, click Predict, expect the setup hint (no stack trace).
    ARTIFACT.rename(ARTIFACT_BACKUP)
    try:
        click_predict(client)
        ok = wait_for(client, 'document.querySelector(".stAlert") !== null', timeout=30)
        print(f"[check] missing-model alert present: {ok}")
        # Confirm no stack trace text leaks into the UI.
        ui_text = client.send("Runtime.evaluate", {
            "expression": "document.body.innerText",
            "returnByValue": True,
        })["result"]["value"]
        has_traceback = "Traceback" in ui_text or "Traceback (most recent call last)" in ui_text
        print(f"[check] stack trace in UI: {has_traceback} (expect False)")
        screenshot(client, "screenshot_missing_model.png", "Missing model artifact -> setup message")
    finally:
        ARTIFACT_BACKUP.rename(ARTIFACT)

    # ---- 3. Successful prediction (defaults == first dataset row) --------
    click_predict(client)
    wait_for(client, 'document.querySelector(".stSuccess") !== null', timeout=60)
    time.sleep(0.5)
    ui_text = client.send("Runtime.evaluate", {
        "expression": "document.body.innerText",
        "returnByValue": True,
    })["result"]["value"]
    import re
    m = re.search(r"\*\*Predicted scaled sound pressure level:\*\*\s*([\d.]+)\s*dB", ui_text)
    print(f"[check] prediction text: {m.group(0) if m else 'NOT FOUND'}")
    screenshot(client, "screenshot_prediction.png", "Successful prediction result (~126.40 dB)")

    # ---- 4. Out-of-range value cannot reach inference --------------------
    # Push frequency above its allowed max (20000 Hz) via the DOM and click
    # Predict. Expect validation to block inference (error shown, no prediction).
    set_input(client, "frequency_hz", "99999")
    click_predict(client)
    time.sleep(1.0)
    ui_text = client.send("Runtime.evaluate", {
        "expression": "document.body.innerText",
        "returnByValue": True,
    })["result"]["value"]
    has_prediction = "Predicted scaled sound pressure level" in ui_text
    print(f"[check] out-of-range reached inference: {has_prediction} (expect False)")
    screenshot(client, "screenshot_validation.png", "Out-of-range value blocked")

    # Restore the input to a valid value so the app is left in a clean state.
    set_input(client, "frequency_hz", "800")
    print("\nAll Section 11.2 acceptance checks complete.")


def click_predict(client):
    client.send("Runtime.evaluate", {
        "expression": """
        (() => {
          const buttons = Array.from(document.querySelectorAll('button'));
          const b = buttons.find(el => el.innerText.trim() === 'Predict');
          if (b) { b.click(); return true; }
          return false;
        })()
        """,
        "returnByValue": True,
        "awaitPromise": True,
    })


def set_input(client, key: str, value: str):
    client.send("Runtime.evaluate", {
        "expression": f"""
        (() => {{
          const inputs = Array.from(document.querySelectorAll('input[type="number"]'));
          // field order: frequency, angle of attack, chord, velocity, thickness
          const idx = ["frequency_hz","angle_of_attack_deg","chord_length_m","free_stream_velocity_m_s","suction_side_displacement_thickness_m"].indexOf("{key}");
          const el = inputs[idx];
          if (!el) return false;
          el.focus();
          el.value = {value};
          el.dispatchEvent(new Event('input', {{bubbles: true}}));
          el.dispatchEvent(new Event('change', {{bubbles: true}}));
          el.blur();
          return true;
        }})()
        """,
        "returnByValue": True,
        "awaitPromise": True,
    })


if __name__ == "__main__":
    main()
