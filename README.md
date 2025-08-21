# STRIDE-QA-mini Object-centric Spatial QA — QLoRA 微調整 & 簡易比較

参考: https://arxiv.org/abs/2508.10427

## 1. 検証環境

Ryzen 9 9950x3D
DDR5-6400MHz 64GB
RTX4080 VRAM 16GB

---

## 2. Dataset取得

```bash
uv run dataset_pre_download.py
```

``` text
data/
  stride-qa-mini/
    images/
      *.jpg
    object_centric_spatial_qa.json
```

---

## 3. 学習（QLoRA）

```bash
python train_objspatial_qlora.py
```

## 4. 単枚比較（Base vs QLoRA）

```bash
uv run compare_one.py \
  --data_json data/stride-qa-mini/object_centric_spatial_qa.json \
  --img_root  data/stride-qa-mini/images \
  --sample_id <id> \
  --lora_path qwen25vl_3b_obj_center_qlora/lora_adapter \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
```

出力例（要旨）：

``` bash
===== Sample =====
ID: <id>
Question: Who is positioned more to the left, Region [2] ... or Region [0] ...?
GT Answer: From the viewer's perspective, Region [2] appears more on the left side.

--- Base ---
<生成テキスト>
Eval: {'type': 'region', 'ok': False, 'gt': 2, 'pred': 0}

--- QLoRA ---
<生成テキスト>
Eval: {'type': 'region', 'ok': True, 'gt': 2, 'pred': 2}

===== Summary =====
Base : NG
QLoRA: OK
```
