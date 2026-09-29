"""Drive the proxy like an agent: N sequential calls in one session, feeding replies back."""
import sys, uuid, httpx

port, n = int(sys.argv[1]), int(sys.argv[2])
sid = str(uuid.uuid4())
msgs = [{"role": "system", "content": "You are a terminal agent."}, {"role": "user", "content": "Task: make the tests pass."}]
for i in range(n):
    r = httpx.post(f"http://127.0.0.1:{port}/chat/completions", headers={"X-Session-ID": sid, "Authorization": "Bearer sk-x"},
                   json={"model": "stayswitch", "messages": msgs}, timeout=60)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]
    print(i, r.json()["model"], reply[:40])
    msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"observation {i}"}]
