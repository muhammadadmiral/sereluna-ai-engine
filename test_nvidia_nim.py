import os
import time
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

def test_nvidia_nim():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not found in .env")
        return

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test faster model for comparison
    model = "meta/llama-3.1-8b-instruct"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are Sereluna, a helpful mental health assistant."},
            {"role": "user", "content": "Halo, apa kabar? Aku merasa sedikit cemas hari ini."}
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 0.95
    }

    print(f"🚀 Sending request to NVIDIA NIM ({model})...")
    start_time = time.perf_counter()
    
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            elapsed = (time.perf_counter() - start_time) * 1000
            
            content = res_data["choices"][0]["message"]["content"]
            print(f"✅ Success! Response time: {elapsed:.2f}ms")
            print(f"🤖 Bot: {content}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_nvidia_nim()
