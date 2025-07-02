# 3D Point Cloud Compression with MPPQNet

## 簡介

本專案實作了一套基於 Moment-Preserving Product Quantization Neural Network (MPPQNet) 的 3D 點雲壓縮框架，包含資料預處理、編碼 (encode)、解碼 (decode)、辭典學習 (dictionary learning) 等模組。

主要功能：
- 特徵抽取 (`feature_extraction.py`)
- 編碼管線 (`encode_pipeline.py`)
- 解碼管線 (`decode_pipeline.py`)
- 字典與高斯模型 (`dictionary_and_gaussian.py`)
- 自訂神經網路結構 (`networks.py`, `mpnn_qnn.py`)
- 壓縮與工具函式 (`compression_utils.py`)
- 整合式字典學習流水線 (`pipeline_dictlearning.py`)

## 環境需求

- Python 3.8+
- 建議使用虛擬環境 (venv 或 conda)

## 安裝

1. 將專案 clone 到本機  

```bash
git clone https://github.com/GGbby/MPPQNet.git
cd <repo-name>
```

2. 在專案根目錄建立並啟動虛擬環境

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
```

3. 安裝相依套件

```bash
pip install -r requirements.txt
```

## 執行範例

特徵抽取

```bash
# 編碼（Encode）
python feature_extraction.py --input path/to/pointcloud.ply --output path/to/features.npz
```

```bash
# 解碼（Decode）
python encode_pipeline.py --config configs/encode_config.yaml
```

```bash
# 字典學習
python decode_pipeline.py --config configs/decode_config.yaml
```

```bash
python pipeline_dictlearning.py --dataset path/to/dataset --epochs 100
```

## 專案結構

```markdown
├── .gitignore
├── README.md
├── requirements.txt
└── MPPQNet/
    ├── feature_extraction.py
    ├── compression_utils.py
    ├── dictionary_and_gaussian.py
    ├── encode_pipeline.py
    ├── decode_pipeline.py
    ├── mpnn_qnn.py
    ├── networks.py
    ├── pipeline_dictlearning.py    
└── dataset/
    └── your data
```
