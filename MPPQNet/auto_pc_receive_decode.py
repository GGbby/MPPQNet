# auto_pc_receive_decode.py
import socket
import pickle
import os
import sys
import glob
from tqdm import tqdm
from threading import Thread
from decode_pipeline import decode_one  # 確保同目錄下有此模組

# 參數
LISTEN_PORT = 9000
BIN_DIR     = 'received_streams_bin'
PLY_OUT     = 'recon_ply_bin'
USE_GAUSS   = True

os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(PLY_OUT, exist_ok=True)

def recv_all(conn, length):
    buf = b''
    while len(buf) < length:
        chunk = conn.recv(length - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def handle_client(conn):
    idx = 0
    while True:
        hdr = conn.recv(4)
        if not hdr:
            break
        length = int.from_bytes(hdr, 'big')
        blob = recv_all(conn, length)
        if blob is None:
            break

        # 存檔
        bin_path = os.path.join(BIN_DIR, f"stream_{idx:03d}.bin")
        with open(bin_path, 'wb') as f:
            f.write(blob)
        print(f"[Received] {bin_path}")

        # 解碼成 .ply
        dummy_bar = tqdm(total=0, bar_format='', position=2, leave=False)
        try:
            decode_one(bin_path, PLY_OUT, dummy_bar, USE_GAUSS)
            print(f"[Decoded ] {bin_path} -> {PLY_OUT}")
        except Exception as e:
            print(f"[Error] decode failed: {e}")
        dummy_bar.close()

        idx += 1

    conn.close()

def server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', LISTEN_PORT))
    sock.listen(1)
    print(f"Listening on port {LISTEN_PORT} ...")
    conn, addr = sock.accept()
    print("Connected by", addr)
    handle_client(conn)
    sock.close()

if __name__ == '__main__':
    server()
