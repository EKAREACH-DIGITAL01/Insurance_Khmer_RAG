import base64
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

from khmer_rag.config import ACTIVE_EMBEDDING_MODEL_DIR, SERVER_CTX_SIZE

OS_NAME = platform.system()

print(f"Detected OS: {OS_NAME}")

LLAMA_SERVER = shutil.which("llama-server")

if LLAMA_SERVER is None:
    if OS_NAME == "Windows":
        raise SystemExit("llama-server.exe not found. Please install llama.cpp and add it to PATH.")

    raise SystemExit("llama-server not found. Please install llama.cpp and add it to PATH.")

print(f"llama-server: {LLAMA_SERVER}")

# The production generator. No longer a menu: SeaLLM v3 7B won the
# four-way comparison in reports/generator_model_comparison/ (combined
# 4.8815, the lowest hallucination count at 5, and zero truncated answers),
# so it is simply what the chatbot runs.
#
# The other three GGUFs (Qwen3.5 9B, Sailor2-L 8B, Llama-3.1 8B) are still
# installable via install_models.py and still benchmarked by
# scripts/13_generate_model_answers.py, which carries its own model table
# including the per-model llama-server flags they need. Nothing here has
# to change to re-run that comparison.
GENERATOR_FILE = "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf"
GENERATOR_NAME = "SeaLLM"

# SeaLLM is not a reasoning model -- --jinja is still passed so the GGUF's
# own embedded chat template is used correctly rather than llama-server's
# built-in heuristic fallback.
extra_args = ["--jinja"]

model = GENERATOR_FILE
model_name = GENERATOR_NAME

os.environ["CHATBOT_MODEL"] = model_name


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "generator" / model

print("""
========================================
Khmer Legal Chatbot
========================================
""")

if not MODEL_PATH.exists():
    raise SystemExit(
        f"Generator model not found:\n  {MODEL_PATH}\n\n"
        f"Run: python install_models.py  (choose {GENERATOR_NAME})"
    )

print(f"Generator model  : {model}")


def active_embedding_model_dir():
    """Use the same retriever setting as the application."""
    return ACTIVE_EMBEDDING_MODEL_DIR


def verify_embedding_model():
    """Confirms the index app.py is about to load exists, and prints which
    embedding model built it.

    There is no embedding-model selection: app.py loads exactly
    vector_db/<ACTIVE_EMBEDDING_MODEL_DIR>/, which BGE-M3 won in
    reports/embedding_model_comparison/. Checking it here means a missing
    or half-built index fails immediately, with the command that fixes it,
    instead of after llama-server has spent a minute loading a 4.7 GB
    model."""
    active_dir = active_embedding_model_dir()
    if not active_dir:
        raise SystemExit(
            "No ACTIVE_EMBEDDING_MODEL_DIR in khmer_rag.config -- "
            "cannot tell which vector_db/ index the app will load."
        )

    db_dir = PROJECT_ROOT / "vector_db" / active_dir
    index = db_dir / "khmer_insurance.index"
    metadata = db_dir / "khmer_insurance_metadata.json"

    missing = [p.name for p in (index, metadata) if not p.exists()]
    if missing:
        raise SystemExit(
            f"\nVector database incomplete: vector_db/{active_dir}/ is missing "
            f"{', '.join(missing)}.\n"
            f"Run: python scripts/08_generate_embeddings.py"
        )

    info_path = db_dir / "embedding_model_info.json"
    label = active_dir
    if info_path.exists():
        try:
            label = json.loads(info_path.read_text(encoding="utf-8")).get("label", active_dir)
        except (json.JSONDecodeError, OSError):
            pass

    print(f"Embedding model  : {label}")
    print(f"Vector database  : vector_db/{active_dir}/")


verify_embedding_model()


def process_name(pid):
    """Best-effort executable name for a pid, or None if it can't be read."""
    try:
        if OS_NAME == "Windows":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            if out.startswith('"'):
                return out.split('","')[0].strip('"')
            return None
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def is_llama_server(name):
    """True if a process name looks like llama.cpp's server binary.

    Matches the bare name and full paths, with or without .exe --
    llama-server, llama-server.exe, /opt/homebrew/bin/llama-server, and
    the older llama.cpp 'server' binary name all count."""
    if not name:
        return False
    base = Path(name.strip().strip('"')).name.lower()
    if base.endswith(".exe"):
        base = base[:-4]
    return base in {"llama-server", "server"} or base.startswith("llama-server")


def free_port_8080():
    """Kill any llama-server already bound to port 8080 before we launch ours.

    A stale server left over from a crashed run, or one started by
    scripts/13_generate_model_answers.py (or a previous chatbot launch), keeps
    the port. Our new process then fails to bind, dies, and the readiness poll
    below silently attaches to the WRONG server -- commonly the bare llama.cpp
    DEFAULT of n_ctx=4096, which then 400s every prompt over 4096 tokens
    ('exceeds the available context size (4096)'). This is exactly what made
    Sailor break. Freeing the port first guarantees we only talk to the server
    we just started.

    Only llama-server processes are killed. Port 8080 is a popular default
    (dev servers, proxies, Tomcat...), and silently SIGKILLing whatever
    unrelated process happens to hold it is not something a chatbot
    launcher should do -- if the holder isn't ours, abort and let the user
    decide."""
    pids = []
    try:
        if OS_NAME == "Windows":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            for line in out.splitlines():
                if ":8080" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        pids.append(parts[-1])
        else:
            out = subprocess.run(
                ["lsof", "-nP", "-iTCP:8080", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            pids = [p for p in out.split() if p.isdigit()]
    except Exception:
        return

    pids = list(dict.fromkeys(pids))  # de-dupe, preserve order
    if not pids:
        return

    foreign = [(pid, process_name(pid)) for pid in pids]
    foreign = [(pid, name) for pid, name in foreign if not is_llama_server(name)]
    if foreign:
        detail = ", ".join(f"pid {pid} ({name or 'unknown'})" for pid, name in foreign)
        raise SystemExit(
            f"\nPort 8080 is held by a process that is not llama-server: "
            f"{detail}.\nRefusing to kill it. Stop that process (or free the "
            f"port) yourself, then re-run."
        )

    print(
        f"WARNING: port 8080 already in use (pid {', '.join(pids)}); killing "
        f"stale server(s) so we don't attach to the wrong context."
    )
    for pid in pids:
        try:
            if OS_NAME == "Windows":
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
            else:
                os.kill(int(pid), 15)
        except Exception:
            pass
    time.sleep(2)
    # Force-kill anything still holding the port (POSIX only; taskkill above is
    # already a force-kill on Windows).
    if OS_NAME != "Windows":
        try:
            still = subprocess.run(
                ["lsof", "-nP", "-iTCP:8080", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.split()
            for pid in still:
                # Re-check the name: by now the port could have been picked
                # up by something unrelated, and SIGKILL is unrecoverable.
                if pid.isdigit() and is_llama_server(process_name(pid)):
                    os.kill(int(pid), 9)
        except Exception:
            pass
        time.sleep(1)


def verify_server_ctx(expected_ctx):
    """Confirm the live server's context window is what we launched with.

    Reads /props; llama.cpp exposes the real n_ctx under
    default_generation_settings.n_ctx (and sometimes top-level). If it's smaller
    than requested, we're talking to the wrong/misconfigured server -- fail
    loudly instead of silently 400-ing every long prompt mid-chat."""
    try:
        props = requests.get("http://localhost:8080/props", timeout=10).json()
    except Exception as e:
        print(f"WARNING: could not read /props to verify n_ctx: {e}")
        return
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
    if n_ctx is None:
        print("WARNING: server did not report n_ctx; could not verify context size.")
        return
    if int(n_ctx) < expected_ctx:
        raise SystemExit(
            f"\nllama-server came up with n_ctx={n_ctx}, but {expected_ctx} was "
            f"requested. This is the wrong/stale server (long prompts will 400 "
            f"with 'exceeds the available context size'). Aborting -- make sure "
            f"no other llama-server is running on port 8080, then re-run."
        )
    print(f"Context size verified: n_ctx={n_ctx}.")


free_port_8080()

# --ctx-size: MUST exceed prompt_token + max_tokens or llama.cpp falls back
# to context-shifting (re-evaluating large chunks of the prompt), which is
# far slower than ordinary prefill -- this was previously unset, silently
# defaulting to a small window (commonly 4096) that the app's ~8k-15k token
# prompts blew straight through. 16384 gives headroom above CONTEXT_K=5's
# typical prompt size (~8-9k tokens) + max_tokens=1024 in app.py, with room
# to spare for longer questions/chunks. Raise further if you see this get
# hit again (e.g. after increasing CONTEXT_K).
#
# --flash-attn: faster attention computation, more noticeable at this
# context length than at a short one.
#
# --cache-type-k / --cache-type-v: quantize the KV cache (FP16 -> Q8_0).
# At ~9-16k tokens the KV cache itself is large enough that this matters
# for both memory headroom and avoiding any swap-induced slowdown.
llama_process = subprocess.Popen(
    [
        LLAMA_SERVER,
        "-m",
        str(MODEL_PATH),
        "-ngl",
        "999",
        "--ctx-size",
        str(SERVER_CTX_SIZE),
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        # 127.0.0.1, not 0.0.0.0: llama-server is an unauthenticated local
        # sidecar and app.py only ever talks to it over localhost:8080
        # (LLAMA_SERVER_URL). Binding all interfaces exposed an open
        # inference endpoint to anyone on the same network for no benefit.
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
    + extra_args
)

print("Waiting for llama-server...")

start_time = time.time()

while True:
    if time.time() - start_time > 120:
        llama_process.kill()
        raise SystemExit("llama-server failed to start within 120 seconds.")

    # If our own process died (e.g. couldn't bind the port, or OOM at this ctx
    # size), stop -- otherwise the poll could latch onto a different server
    # still listening on 8080.
    if llama_process.poll() is not None:
        raise SystemExit(
            f"llama-server process exited during startup "
            f"(code {llama_process.returncode}). Check the model loads at "
            f"--ctx-size {SERVER_CTX_SIZE} and that port 8080 is free."
        )

    try:
        response = requests.get(
            "http://localhost:8080/v1/models",
            timeout=5,
        )

        if response.status_code == 200:
            break

    except Exception:
        pass

    time.sleep(1)

verify_server_ctx(SERVER_CTX_SIZE)

print("Model loaded.")


def patch_streamlit_browser_chrome(title, favicon_path):
    """Sets the browser tab's title/favicon to match this app instead of
    Streamlit's defaults, by patching the static index.html Streamlit
    serves (st.set_page_config() in app.py can't do this on its own --
    it only takes effect once the page has already loaded and the
    websocket has connected, by which point the static HTML's own
    title/icon have already been read once).

    Patches whichever streamlit installation `import streamlit` resolves
    to for the interpreter running this script. The server below is
    launched via `sys.executable -m streamlit` specifically so it's
    guaranteed to be that exact same installation -- a bare "streamlit"
    command resolves via PATH instead, which could silently pick a
    different installed copy on a machine with more than one Python/
    Streamlit install.

    Safe to run on every launch -- always strips any previous version of
    this patch first (in case an older, different patch logic was applied
    before) and reapplies cleanly, so it can never accumulate stale
    leftovers across app updates or get stuck skipping a real fix.

    Note: a brief "Streamlit" flash specifically on the very first
    navigation to a freshly-opened tab (but never on a reload of an
    already-open one) is a Chromium tab-title-from-history quirk, not
    something page content can control -- not worth chasing further."""
    try:
        import streamlit as st_pkg

        index_path = Path(st_pkg.__file__).parent / "static" / "index.html"
        if not index_path.exists():
            return

        html = index_path.read_text(encoding="utf-8")

        # Strip any title-enforcer <script> block and marker comment left
        # behind by earlier, more complex versions of this patch, so this
        # always starts from a clean slate instead of layering a new
        # marker on top while the old (now-unwanted) script stays behind.
        html = re.sub(
            r"<script>\s*\(function\(\) \{.*?khmer-chatbot.*?\}\)\(\);\s*</script>\s*",
            "",
            html,
            flags=re.DOTALL,
        )
        html = re.sub(r"<!--\s*khmer-chatbot-branding-patch[^>]*-->\s*", "", html)

        marker = "<!-- khmer-chatbot-branding-patch -->"

        favicon_bytes = Path(favicon_path).read_bytes()
        favicon_b64 = base64.b64encode(favicon_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{favicon_b64}"

        patched, title_count = re.subn(
            r"<title>.*?</title>", f"<title>{title}</title>", html, count=1
        )
        patched, icon_count = re.subn(
            r'<link rel="(?:shortcut icon|icon)"[^>]*>',
            f'<link rel="icon" href="{data_uri}">',
            patched,
            count=1,
        )

        if title_count == 0 or icon_count == 0:
            # Expected tags weren't found (unexpected streamlit static
            # layout for this version) -- don't write a no-op/partial patch.
            return

        patched = patched.replace("</head>", f"{marker}\n</head>", 1)
        index_path.write_text(patched, encoding="utf-8")
    except Exception:
        # Never let a cosmetic patch block the actual app from starting.
        pass


patch_streamlit_browser_chrome(
    "Khmer Insurance Law Assistant",
    PROJECT_ROOT / "assets" / "logo" / "favicon.png",
)

print("Launching Streamlit...")

# Using `sys.executable -m streamlit` (not the bare "streamlit" command) so
# this is the exact same installation that was just patched above.
subprocess.run(
    [
        sys.executable,
        "-m",
        "streamlit",
        "cache",
        "clear",
    ]
)

try:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "app.py"),
            "--client.toolbarMode=minimal",
            "--browser.gatherUsageStats=false",
        ]
    )
finally:
    llama_process.terminate()
