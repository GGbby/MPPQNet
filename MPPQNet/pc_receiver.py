# pc_receiver_v2_bin.py
# PC 端：接收來自 Jetson Nano 的 .bin bit-stream，緩存字典並儲存為 .bin 檔

import socket
import pickle
import os
from tqdm import tqdm

# 監聽埠號
LISTEN_PORT = 9000
# 儲存接收的原始 bitstream
SAVE_DIR = 'received_streams_bin'
# 緩存首批字典資料
DICT_CACHE = {}


def recv_all(conn, length):
    """
    確保從 conn 中接收 length bytes
    """
    buf = b''
    while len(buf) < length:
        chunk = conn.recv(length - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def handle_client(conn):
    os.makedirs(SAVE_DIR, exist_ok=True)
    idx = 0
    while True:
        # 讀取 4 字節長度標頭
        hdr = conn.recv(4)
        if not hdr:
            break
        length = int.from_bytes(hdr, 'big')
        # 接收整段 blob
        blob = recv_all(conn, length)
        if blob is None:
            break
        # 反序列化
        payload = pickle.loads(blob)

        # 若包含字典，緩存到全域 DICT_CACHE
        if 'gauss_mu_pf' in payload:
            DICT_CACHE['gauss_mu_pf']    = payload['gauss_mu_pf']
            DICT_CACHE['gauss_sigma_pf'] = payload['gauss_sigma_pf']
            DICT_CACHE['pt_centers']     = payload['pt_centers']
            DICT_CACHE['res_centers']    = payload['res_centers']
            DICT_CACHE['gauss_mu_res']   = payload['gauss_mu_res']
            DICT_CACHE['gauss_sigma_res']= payload['gauss_sigma_res']

        # 將原始 blob 儲存為 .bin
        out_path = os.path.join(SAVE_DIR, f"stream_{idx:03d}.bin")
        with open(out_path, 'wb') as f:
            f.write(blob)
        idx += 1

    conn.close()


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', LISTEN_PORT))
    sock.listen(1)
    print(f"Listening on port {LISTEN_PORT}...")
    conn, addr = sock.accept()
    print("Connected by", addr)
    handle_client(conn)
    sock.close()

if __name__ == '__main__':
    main()
