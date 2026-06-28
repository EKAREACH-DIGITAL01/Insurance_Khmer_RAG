import base64
import json
import re
import subprocess
import sys
import time
import requests
import platform
import shutil
import os
from pathlib import Path

OS_NAME = platform.system()

print(f"Detected OS: {OS_NAME}")

LLAMA_SERVER = shutil.which("llama-server")

if LLAMA_SERVER is None:

    if OS_NAME == "Windows":
        raise SystemExit(
            "llama-server.exe not found. "
            "Please install llama.cpp and add it to PATH."
        )

    raise SystemExit(
        "llama-server not found. " "Please install llama.cpp and add it to PATH."
    )

print(f"llama-server: {LLAMA_SERVER}")

MODELS = {
    "1": {
        "file": "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf",
        "name": "SeaLLM",
        # Not a reasoning model -- --jinja is still added so the GGUF's
        # own embedded chat template is used correctly rather than
        # llama-server's built-in heuristic fallback.
        "extra_args": ["--jinja"],
    },
    "2": {
        "file": "Qwen3.5-9B-Q4_K_M.gguf",
        "name": "Qwen",
        # Qwen3.5 is a hybrid-reasoning model: by default it wraps a
        # <think>...</think> block before its actual answer. Without
        # --jinja, llama-server can't parse the GGUF's embedded chat
        # template at all and the <think> tags leak straight into
        # delta.content as plain text. Without --reasoning off, the model
        # still thinks first -- and on this corpus's long (~8-9k token)
        # RAG prompts, thinking alone can consume the entire max_tokens=1024
        # budget in app.py, so the real answer never gets generated and
        # delta.content stays empty for the whole stream (sources/
        # disclaimer still render in app.py since those don't depend on
        # answer_text being non-empty -- this is what looked like a
        # "silent success" with no visible answer in the UI).
        # --reasoning off is a startup flag (no per-request override
        # exists in llama-server yet), so it's set here once for the
        # whole server process rather than per-request.
        "extra_args": ["--jinja", "--reasoning", "off"],
    },
    "3": {
        # Sailor2-L = the 128K-context variant. The plain Sailor2-8B GGUF is
        # capped at 4096 ctx -- too small for our 5k-13k-token RAG prompts, so
        # it 400'd on nearly every question. Sailor2-L is natively 131072 ctx.
        "file": "Sailor2-L-8B-Chat-Q4_K_M.gguf",
        "name": "Sailor2-L",
        # Sailor2 8B is Qwen2.5-based (ChatML template) and SEA-tuned
        # (incl. Khmer), so it's a fair-spec peer to SeaLLM/Qwen/Llama --
        # unlike Gemma 4 E4B, which was an effective-~4B on-device variant
        # with thin Khmer coverage and underperformed badly. --jinja parses
        # the GGUF chat template; no reasoning-channel mechanism here.
        "extra_args": ["--jinja"],
    },
    "4": {
        "file": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "name": "Llama",
        # Not a reasoning model -- no <think> or <|channel|> mechanism at
        # all in this architecture. --jinja still added for consistency
        # and correctness with the other three.
        "extra_args": ["--jinja"],
    },
}

print("""
========================================
Khmer Legal Chatbot
========================================

Choose model:

1. SeaLLMs-v3-7B-Chat-Q4_K_M.gguf
2. Qwen3.5-9B-Q4_K_M.gguf
3. Sailor2-L-8B-Chat-Q4_K_M.gguf
4. Llama-3.1-8B-Instruct-Q4_K_M.gguf
""")

choice = input("Choice: ").strip()

if choice not in MODELS:
    raise SystemExit("Invalid choice.")

selected = MODELS[choice]

model = selected["file"]
model_name = selected["name"]
extra_args = selected["extra_args"]

os.environ["CHATBOT_MODEL"] = model_name


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "generator" / model

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found:\n{MODEL_PATH}")

print(f"\nStarting {model}...")


def print_embedding_model_status():
    """Reads vector_db/embedding_model_info.json (written by
    scripts/07_generate_embeddings.py) and prints which embedding model
    actually built the index this session will query against -- the live
    chatbot is only correct if it loads the SAME model app.py's own
    load_embedding_model_info() resolves to, so surfacing this here gives
    a single place to catch a mismatch before it shows up as confusingly
    bad retrieval quality.

    Mirrors app.py's own fallback: if the file doesn't exist (a
    pre-existing vector_db/ built before multiple embedding models were
    supported), assumes BGE-M3, since that was the only option that ever
    existed before this file did."""
    info_path = PROJECT_ROOT / "vector_db" / "embedding_model_info.json"

    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            print(f"Embedding model  : {info.get('label', 'unknown')}")
            return
        except (json.JSONDecodeError, OSError) as e:
            print(f"Embedding model  : could not read {info_path} ({e})")
            return

    print(
        "Embedding model  : BAAI/bge-m3 (assumed -- no "
        "vector_db/embedding_model_info.json found; this is expected for "
        "a vector_db/ built before scripts/07_generate_embeddings.py "
        "supported choosing a model)"
    )


print_embedding_model_status()


SERVER_CTX_SIZE = 16384  # must match app.py's SERVER_CTX_SIZE


def free_port_8080():
    """Kill any llama-server already bound to port 8080 before we launch ours.

    A stale server left over from a crashed run, or one started by
    scripts/11_compare_generator_models.py (or a previous chatbot launch), keeps
    the port. Our new process then fails to bind, dies, and the readiness poll
    below silently attaches to the WRONG server -- commonly the bare llama.cpp
    DEFAULT of n_ctx=4096, which then 400s every prompt over 4096 tokens
    ('exceeds the available context size (4096)'). This is exactly what made
    Sailor break. Freeing the port first guarantees we only talk to the server
    we just started."""
    pids = []
    try:
        if OS_NAME == "Windows":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            for line in out.splitlines():
                if ":8080" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        pids.append(parts[-1])
        else:
            out = subprocess.run(
                ["lsof", "-nP", "-iTCP:8080", "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            pids = [p for p in out.split() if p.isdigit()]
    except Exception:
        return

    pids = list(dict.fromkeys(pids))  # de-dupe, preserve order
    if not pids:
        return

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
                capture_output=True, text=True, timeout=10,
            ).stdout.split()
            for pid in still:
                if pid.isdigit():
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
    n_ctx = (
        props.get("default_generation_settings", {}).get("n_ctx")
        or props.get("n_ctx")
    )
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
        "--host",
        "0.0.0.0",
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
    PROJECT_ROOT / "assets" / "favicon.png",
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