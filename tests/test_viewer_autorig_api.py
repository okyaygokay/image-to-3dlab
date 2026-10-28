
"""The Animate tab's HTTP surface, driven the way the page drives it."""

from __future__ import annotations

import json
from pathlib import Path

# --- the frontend path: POST /api/animate/describe through the real handler ------------------

def test_the_describe_endpoint_takes_a_prompt_and_starts_the_colab_job(
        tmp_path, monkeypatch):
    """The Animate tab's prompt box drives this: a rigged model plus a description. The
    endpoint checks the Colab sign-in (there is no local video model, ever), and the job
    it starts is one supervisor run of the prompt2rig backend with the prompt forwarded."""
    import http.server
    import sys
    import threading
    import urllib.error
    import urllib.request

    viewer = Path(__file__).resolve().parents[1] / "viewer"
    if str(viewer) not in sys.path:
        sys.path.insert(0, str(viewer))
    import autorig_api
    import generate_api as api

    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    run.mkdir(parents=True)
    rigged = run / "knight_rigged.glb"
    rigged.write_bytes(b"glTF")
    monkeypatch.setattr(api, "resolve_animate_model", lambda _path, _root: rigged)
    monkeypatch.setattr(api, "autorig_colab_ready", lambda: True)
    monkeypatch.setattr(api, "OUTPUT_ROOT", tmp_path)

    started: list = []

    def fake_runner(job, _manager=None):
        started.append(job)
        job.status = "done"

    monkeypatch.setattr(api, "run_animate_job", fake_runner)
    monkeypatch.setattr(api, "ANIMATE_JOBS", autorig_api.AnimateJobManager(
        tmp_path / "animate", tmp_path / "no-motions"))

    def post(body: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            "http://127.0.0.1:1/api/animate/describe",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     # the studio page's own request, as request_utils.check_request wants
                     "X-AssetFurnace-Request": "studio"})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            request.full_url = f"http://127.0.0.1:{server.server_port}/api/animate/describe"
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as exc:
                return exc.code, json.load(exc)
        finally:
            server.shutdown()
            server.server_close()

    status, body = post({"model": "animate/knight__rig__20261005-120000/knight_rigged.glb",
                         "prompt": "swinging a sword"})
    assert status == 202 and len(started) == 1
    assert body["events_url"] == f"/api/animate/{started[0].id}/events"
    assert started[0].record["motion"]["prompt"] == "swinging a sword"

    monkeypatch.setattr(api, "autorig_colab_ready", lambda: False)
    status, body = post({"model": "x", "prompt": "swinging a sword"})
    assert status == 409 and "sign in" in body["error"].lower()

    monkeypatch.setattr(api, "autorig_colab_ready", lambda: True)
    status, body = post({"model": "animate/knight__rig__20261005-120000/knight_rigged.glb",
                         "prompt": "  "})
    assert status == 422
