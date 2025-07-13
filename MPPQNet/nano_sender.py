# nano_sender_v2_bin.py
# Jetson Nano 端：以固定頻率傳送 .bin bit-stream，前 INCLUDE_DICT_COUNT 筆包含完整字典，之後僅傳 codes

import time
import socket
import glob
import pickle
import os

# 伺服器設定
SERVER_IP = '192.168.50.127'
SERVER_PORT = 9000
# bit-stream 資料夾（encode_pipeline_v2_bin.py 輸出）
STREAM_DIR = 'bitstreams_bin'
# 傳送頻率 (Hz)
FREQUENCY = 1.0
# 前幾筆需包含殘差字典
INCLUDE_DICT_COUNT = 5


def send_stream(sock, bin_path, include_dict):
    """
    從 .bin 檔讀取 payload，根據 include_dict 決定是否包含字典欄位，
    然後 pickle 序列化並送出（長度前綴 + blob）。
    """
    # 讀取原始 bit-stream dict
    with open(bin_path, 'rb') as f:
        payload = pickle.load(f)

    # 構造傳送用 dict
    to_send = {
        'pf_codewords': payload['pf_codewords'],
        'pf_thr':       payload['pf_thr'],
        'pt_labels':    payload['pt_labels'],
        'res_labels':   payload['res_labels']
    }
    if include_dict:
        # 加入完整字典欄位
        for key in [
            'gauss_mu_pf', 'gauss_sigma_pf',
            'pt_centers', 'res_centers',
            'gauss_mu_res', 'gauss_sigma_res'
        ]:
            to_send[key] = payload[key]

    # pickle 序列化
    blob = pickle.dumps(to_send)
    # 傳送 4 字節長度前綴 + blob
    sock.sendall(len(blob).to_bytes(4, 'big') + blob)


def main():
    # 建立 TCP 連線到 PC 端
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))

    # 依檔名排序，逐一傳送
    files = sorted(glob.glob(os.path.join(STREAM_DIR, '*_stream.bin')))
    for idx, path in enumerate(files):
        include = (idx < INCLUDE_DICT_COUNT)
        send_stream(sock, path, include)
        time.sleep(1.0 / FREQUENCY)

    sock.close()

if __name__ == '__main__':
    main()
