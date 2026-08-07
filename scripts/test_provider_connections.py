import os
import sys
import json
import time
import urllib.request
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

def get_env_var(*names):
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None

def test_groq():
    key = get_env_var("GROQ_API_KEY")
    if not key: return False, "Missing GROQ_API_KEY"
    try:
        t0 = time.time()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps({"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Hi"}]}).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, f"Success ({int((time.time()-t0)*1000)}ms): {data['choices'][0]['message']['content'][:40].encode('ascii', 'ignore').decode('ascii')}"
    except Exception as e:
        return False, str(e)

def test_openrouter():
    key = get_env_var("OPENROUTER_API_KEY", "OPenRouter_API_Key")
    if not key: return False, "Missing OPENROUTER_API_KEY"
    try:
        t0 = time.time()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({"model": "meta-llama/llama-3.3-70b-instruct", "messages": [{"role": "user", "content": "Hi"}]}).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "https://veriproof.ai", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, f"Success ({int((time.time()-t0)*1000)}ms): {data['choices'][0]['message']['content'][:40].encode('ascii', 'ignore').decode('ascii')}"
    except Exception as e:
        return False, str(e)

def test_nvidia():
    key = get_env_var("NVIDIA_API_KEY", "NVIDIA_NIM_API_Key")
    if not key: return False, "Missing NVIDIA_API_KEY"
    models_to_try = [
        "meta/llama-3.3-70b-instruct",
        "nvidia/nemotron-4-340b-instruct",
        "meta/llama-3.1-405b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct"
    ]
    for model_name in models_to_try:
        try:
            t0 = time.time()
            req = urllib.request.Request(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                data=json.dumps({"model": model_name, "messages": [{"role": "user", "content": "Hi"}]}).encode("utf-8"),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data['choices'][0]['message']['content'][:40].encode('ascii', 'ignore').decode('ascii')
                return True, f"Success ({model_name}, {int((time.time()-t0)*1000)}ms): {text}"
        except Exception as e:
            last_err = str(e)
    return False, f"All models failed: {last_err}"

def test_mistral():
    key = get_env_var("MISTRAL_API_KEY", "Mistal_API_Key")
    if not key: return False, "Missing MISTRAL_API_KEY"
    try:
        t0 = time.time()
        req = urllib.request.Request(
            "https://api.mistral.ai/v1/chat/completions",
            data=json.dumps({"model": "mistral-small-latest", "messages": [{"role": "user", "content": "Hi"}]}).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, f"Success ({int((time.time()-t0)*1000)}ms): {data['choices'][0]['message']['content'][:40].encode('ascii', 'ignore').decode('ascii')}"
    except Exception as e:
        return False, str(e)

def test_cohere():
    key = get_env_var("COHERE_API_KEY", "Cohere_API_Key")
    if not key: return False, "Missing COHERE_API_KEY"
    try:
        t0 = time.time()
        req = urllib.request.Request(
            "https://api.cohere.com/v2/chat",
            data=json.dumps({"model": "command-r-plus", "messages": [{"role": "user", "content": "Hi"}]}).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", [{}])[0].get("text", "")
            return True, f"Success ({int((time.time()-t0)*1000)}ms): {content[:40].encode('ascii', 'ignore').decode('ascii')}"
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    print("=========================================================================")
    print("   VERIPROOF MULTI-PROVIDER CONNECTIVITY AUDIT                           ")
    print("=========================================================================\n")
    
    providers = [
        ("Groq Cloud", test_groq),
        ("OpenRouter", test_openrouter),
        ("NVIDIA NIM", test_nvidia),
        ("Mistral AI", test_mistral),
        ("Cohere", test_cohere)
    ]
    
    for name, test_fn in providers:
        ok, msg = test_fn()
        status_str = "READY" if ok else "FAILED"
        dots = "." * (25 - len(name))
        print(f"{name} {dots} {status_str} | {msg}")
