"""Section 11.2 frontend acceptance checks via Streamlit AppTest.

Drives the real app code through Streamlit's own widget protocol (not raw DOM),
so widget value changes and button clicks are faithfully round-tripped to the
server. Runs everything in a single AppTest session except the missing-artifact
scenario, which is executed in a fresh subprocess to isolate global session
state. No application code is modified.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parent
APP = REPO / "app" / "streamlit_app.py"
ARTIFACT = REPO / "models" / "best_model.joblib"
ARTIFACT_BACKUP = REPO / "models" / ".best_model.joblib.apptest_backup"

FIRST_ROW = {
    "frequency_hz": 800.0,
    "angle_of_attack_deg": 0.0,
    "chord_length_m": 0.3048,
    "free_stream_velocity_m_s": 71.3,
    "suction_side_displacement_thickness_m": 0.00266337,
}

FIELD_META = [
    ("Frequency", "frequency_hz", "Hz", (200.0, 20000.0)),
    ("Angle of attack", "angle_of_attack_deg", "degrees", (0.0, 22.2)),
    ("Chord length", "chord_length_m", "m", (0.0254, 0.3048)),
    ("Free-stream velocity", "free_stream_velocity_m_s", "m/s", (31.7, 71.3)),
    ("Suction-side displacement thickness", "suction_side_displacement_thickness_m", "m", (0.000400682, 0.0584113)),
]

THREAD_LIMITS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def fresh_at():
    return AppTest.from_file(str(APP)).run(timeout=120)


def main():
    results = []
    at = fresh_at()
    nis = at.number_input

    # ---- 0. Exactly one Predict button; five inputs ----------------------
    buttons = [b for b in at.button]
    results.append(("exactly one 'Predict' button", len(buttons) == 1
                    and [b.label for b in buttons] == ["Predict"],
                    f"{len(buttons)} buttons: {[b.label for b in buttons]}"))
    results.append(("five number inputs", len(nis) == 5, f"{len(nis)} inputs"))

    # ---- 1. Labels, units, allowed ranges --------------------------------
    for i, (label, key, unit, (lo, hi)) in enumerate(FIELD_META):
        el = nis[i]
        ok = label.lower() in str(el.label).lower() and unit in str(el.label)
        results.append((f"input {i}: label='{label}' unit='{unit}'", ok, f"label={el.label!r}"))

    # ---- 2. No prediction before submission ------------------------------
    md = "\n".join(str(m.value) for m in at.markdown)
    results.append(("no prediction before submission",
                    "Predicted scaled sound pressure level" not in md, md[:100]))

    # ---- 3. First real dataset row -> approx 126.40 dB -------------------
    for i, ((_label, key, _unit, _bounds), value) in enumerate(zip(FIELD_META, FIRST_ROW.values())):
        nis[i].set_value(value)
    at.run()
    at.button[0].click().run()
    md = "\n".join(str(m.value) for m in at.markdown)
    m = re.search(r"Predicted scaled sound pressure level:\*\*\s*([\d.]+)\s*dB", md)
    pred = m.group(1) if m else None
    results.append(("first row -> ~126.40 dB", pred == "126.40", f"prediction={pred}"))
    results.append(("exactly one 2-decimal prediction",
                    md.count("Predicted scaled sound pressure level") == 1
                    and pred is not None and len(pred.split(".")[1]) == 2, md))

    # ---- 4. Out-of-range value cannot reach inference --------------------
    # The number_input widget enforces FEATURE_BOUNDS (min_value / max_value),
    # so an out-of-range value cannot even be entered: AppTest's set_value is
    # rejected and the widget keeps an in-range value. Backend validation
    # (pytest test_below_minimum_rejected / test_above_maximum_rejected, already
    # passing) is the second layer that blocks inference.
    hi_freq = FIELD_META[0][3][1]  # frequency max == 20000
    nis[0].set_value(99999.0)  # far above the widget max
    at.run()
    blocked = float(at.number_input[0].value) <= hi_freq
    results.append(("out-of-range blocked by widget (cannot reach inference)",
                    blocked, f"frequency value after out-of-range set={at.number_input[0].value}"))

    # ---- 5. No unapproved Release 2.0 features ---------------------------
    full_ui = md + "\n".join(str(e.value) for e in at.error)
    results.append(("no Release 2.0 features", "2.0" not in full_ui and "Release 2" not in full_ui, "checked"))

    # ---- 6. Missing model artifact -> clear setup message (subprocess) ----
    results.extend(run_missing_model_check())

    # ---- Report ----------------------------------------------------------
    print("SECTION 11.2 MANUAL FRONTEND ACCEPTANCE (AppTest against the app)\n")
    all_pass = True
    for name, ok, detail in results:
        all_pass = all_pass and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\nOVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


def run_missing_model_check():
    """Fresh subprocess: artifact missing, click Predict, expect setup hint,
    no stack trace in the UI."""
    ARTIFACT.rename(ARTIFACT_BACKUP)
    try:
        code = (
            "from streamlit.testing.v1 import AppTest\n"
            "at = AppTest.from_file('app/streamlit_app.py').run(timeout=120)\n"
            "at.button[0].click().run()\n"
            "md = '\\n'.join(str(x.value) for x in at.markdown)\n"
            "errs = '\\n'.join(str(x.value) for x in at.error)\n"
            "excs = [str(e) for e in at.exception]\n"
            "print('MARKDOWN::' + md)\n"
            "print('ERRORS::' + errs)\n"
            "print('EXCEPTIONS::' + '|'.join(excs))\n"
        )
        env = dict(os.environ)
        env.update(THREAD_LIMITS)
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=str(REPO), env=env, timeout=180,
        )
        out = completed.stdout
        md = errs = excs = ""
        for line in out.splitlines():
            if line.startswith("MARKDOWN::"):
                md = line[len("MARKDOWN::"):]
            elif line.startswith("ERRORS::"):
                errs = line[len("ERRORS::"):]
            elif line.startswith("EXCEPTIONS::"):
                excs = line[len("EXCEPTIONS::"):]
        setup_ok = "Run the training workflow" in errs or "Run the training workflow" in md
        no_trace = ("Traceback" not in excs) and ("Traceback" not in md) and ("Traceback" not in errs)
        return ("missing model -> clear setup message", setup_ok, f"errors={errs[:120]}"), \
               ("no stack trace in UI", no_trace, f"exceptions={excs[:120]}")
    finally:
        ARTIFACT_BACKUP.rename(ARTIFACT)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
