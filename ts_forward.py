#!/usr/bin/env python3
"""TCP forwarder — expose MiniLLM on a Tailscale/LAN interface.

usage: ts_forward.py <listen_host> <listen_port> [target_host] [target_port]
   ex: ts_forward.py 100.x.y.z 11435            # tailscale IP -> 127.0.0.1:11435
       ts_forward.py 0.0.0.0 11435              # LAN-wide (careful: no auth!)
"""
import socket, threading, sys

LHOST, LPORT = sys.argv[1], int(sys.argv[2])
RHOST = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
RPORT = int(sys.argv[4]) if len(sys.argv) > 4 else LPORT

def pump(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

def handle(c):
    try:
        u = socket.create_connection((RHOST, RPORT))
    except OSError:
        c.close()
        return
    threading.Thread(target=pump, args=(c, u), daemon=True).start()
    pump(u, c)

if len(sys.argv) < 3:
    sys.exit(__doc__)

srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((LHOST, LPORT))
srv.listen(64)
print(f"forward {LHOST}:{LPORT} -> {RHOST}:{RPORT}", flush=True)
while True:
    c, _ = srv.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
