"""Exercise the installed executable from another directory against a local HTTP fixture.

Also usable without pytest: python tests/test_process.py /absolute/path/to/jcol
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def smoke(command):
    states = []
    slow = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            states.append(body["state"])
            if slow.is_set() and len(states) > 1:
                time.sleep(2)
            answers = {qid: {"noul": 0.9 if "fraud" in body["state"] else 0.1} for qid in body["questions"]}
            payload = json.dumps({"answers": answers, "model": "fixture", "usage": {"cost": 0.0001}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="jcol-process-") as directory:
            root = Path(directory)
            env = {key: value for key, value in os.environ.items()
                   if key not in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL", "PYTHONPATH")}
            env.update(OPENROUTER_API_KEY="fixture-only", JEV_API="openrouter", JEV_MODEL="fixture",
                       JEV_URL=f"http://127.0.0.1:{server.server_port}/decisions",
                       XDG_CONFIG_HOME=str(root / "config"), XDG_CACHE_HOME=str(root / "cache"))

            def run(*args, expected=0, input=None, environment=None):
                result = subprocess.run([str(command), *args], cwd=root, env=environment or env,
                                        input=input, capture_output=True, text=True, timeout=20)
                assert result.returncode == expected, (args, result.returncode, result.stdout, result.stderr)
                return result

            assert "run" in run("--help").stdout
            run("browse", "--help")
            run("init", "--input", "text", "--name", "fraud", "--definition", "Alleges fraud", "-o", "book.json")
            source = "id,text\n1,fraud\n2,ordinary\n3,other\n"
            (root / "source.csv").write_text(source)
            plan = run("run", "source.csv", "--codebook", "book.json", "--dry-run", "--json")
            assert json.loads(plan.stdout)["data"]["cells"] == 3 and not states
            args = ("run", "-", "--input-format", "csv", "--codebook", "book.json", "--project", "run.sqlite",
                    "--no-cache", "--concurrency", "1", "-o", "-", "--quiet", "--json")
            partial = run(*args, "--budget", "0.0001", expected=2, input=source)
            assert len(partial.stdout.splitlines()) == 3
            assert not json.loads(partial.stderr)["data"]["run"]["complete"] and len(states) == 1
            finished = run(*args, input=source)
            assert len(states) == 3 and json.loads(finished.stderr)["data"]["run"]["complete"]
            assert run(*args, input=source).stdout == finished.stdout and len(states) == 3
            no_key = {key: value for key, value in env.items() if key != "OPENROUTER_API_KEY"}
            doctor = run("--json", "doctor", environment=no_key)
            assert not json.loads(doctor.stdout)["data"]["ready"]
            exported = run("export", "run.sqlite", "--source", "source.csv", "-o", "-", "--quiet", environment=no_key)
            assert exported.stdout == finished.stdout and len(states) == 3
            run("export", "run.sqlite", "--source", "source.csv", "-o", "out.parquet", environment=no_key)
            binary = subprocess.run([str(command), "inspect", "-", "--input-format", "parquet", "--json"],
                                    cwd=root, env=no_key, input=(root / "out.parquet").read_bytes(), capture_output=True, timeout=20)
            assert binary.returncode == 0 and json.loads(binary.stdout)["data"]["rows"] == 3

            # SIGINT after a persisted cell: prove actual process recovery, not only an exception handler.
            states.clear()
            slow.set()
            interrupted_args = ("run", "source.csv", "--codebook", "book.json", "--project", "interrupt.sqlite",
                                "--no-cache", "--concurrency", "1", "-o", "interrupted.csv", "--json")
            process = subprocess.Popen([str(command), *interrupted_args], cwd=root, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 10
                while len(states) < 2 and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert len(states) >= 2
                status = json.loads(run("status", "interrupt.sqlite").stdout)
                assert status["columns"][0]["filled"] >= 1
                process.send_signal(signal.SIGINT)
                out, err = process.communicate(timeout=10)
                assert process.returncode == 130 and json.loads(out)["error"]["code"] == "interrupted", (out, err)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            first = states[0]
            states.clear()
            slow.clear()
            run(*interrupted_args)
            assert first not in states
            assert json.loads(run("status", "interrupt.sqlite").stdout)["complete"]
    finally:
        server.shutdown()
        server.server_close()


def test_installed_executable():
    smoke(Path(sys.executable).with_name("jcol"))


if __name__ == "__main__":
    smoke(Path(sys.argv[1]).resolve())
    print("Installed CLI smoke passed: pipes, partial/resume, offline export, Parquet stdin, SIGINT recovery.")
