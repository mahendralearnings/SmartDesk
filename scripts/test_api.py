# scripts/test_api.py
import httpx

BASE = "http://localhost:8000"

def test_health():
    r = httpx.get(f"{BASE}/health")
    print("HEALTH:", r.json())

def test_simple_chat():
    payload = {
        "message": "What is an invoice?",
        "history": [],
        "temperature": 0.7,
    }
    r = httpx.post(f"{BASE}/chat", json=payload, timeout=60)
    data = r.json()
    print(f"\nSIMPLE CHAT [{data['latency_ms']}ms]:")
    print(f"  → {data['reply'][:200]}")

def test_multi_turn():
    history = [
        {"role": "user", "content": "What is an invoice?"},
        {"role": "assistant", "content": "An invoice is a document requesting payment."},
    ]
    payload = {
        "message": "Give me an example of invoice fields.",
        "history": history,
        "temperature": 0.5,
    }
    r = httpx.post(f"{BASE}/chat", json=payload, timeout=60)
    data = r.json()
    print(f"\nMULTI-TURN [{data['latency_ms']}ms]:")
    print(f"  → {data['reply'][:200]}")

def test_validation_error():
    """Temperature out of range → should get 422"""
    payload = {"message": "hi", "temperature": 99.0}
    r = httpx.post(f"{BASE}/chat", json=payload, timeout=10)
    print(f"\nVALIDATION ERROR (expect 422): {r.status_code}")
    print(f"  → {r.json()['detail'][0]['msg']}")

if __name__ == "__main__":
    test_health()
    test_simple_chat()
    test_multi_turn()
    test_validation_error()