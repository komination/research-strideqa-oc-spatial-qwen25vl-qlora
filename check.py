# -*- coding: utf-8 -*-
"""
BaseモデルとLoRA適用後の出力を比較する推論スクリプト。
- 学習時と同じ画像前処理（全景/ハイライト/クロップ/BBOXテキスト）でメッセージを構築
- Base → 推論 → 解放 → LoRA適用 → 推論 → 解放（逐次実行でVRAMを共存させない）
- データセットの「最後のhuman」質問に対して、Base/LoRA/GTを並べて表示
"""

import os
import re
import gc
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from peft import PeftModel
from pycocotools import mask as coco_mask

# ========= ユーザ設定（環境に合わせて変更） =========
IMG_ROOT   = "data/stride-qa-mini/images"
ANN_PATH   = "data/stride-qa-mini/object_centric_spatial_qa.json"
ADAPTER_DIR = "qwen25vl_3b_obj_center_qlora/lora_adapter"  # 学習で保存したLoRA
MODEL_NAME  = "Qwen/Qwen2.5-VL-3B-Instruct"

# 画像キャッシュ
CACHE_DIR     = None  # Noneなら IMG_ROOT/_cache_resize
REBUILD_CACHE = False

# 入力長・画像解像度
MAX_TEXT_LEN = 4096
MIN_PIXELS   = 256 * 28 * 28
MAX_PIXELS   = 512 * 28 * 28  # OOMなら小さく

# 生成パラメータ
GEN_KW = dict(
    max_new_tokens=128,
    do_sample=False,   # 検証用はdeterministic推奨
    temperature=0.0,
    top_p=0.9,
)

# 複数Regionの色（ハイライト）
COLORS = [
    (0, 180, 255),   # R1: 青
    (255, 80, 80),   # R2: 赤
    (120, 220, 120), # R3: 緑
    (255, 200, 0),   # R4: 黄
]

# =====================================================
REGION_PAT_ALL = re.compile(r"Region\s*\[(\d+)\]")

def rle_to_mask(rle_obj: Dict[str, Any]) -> Optional[np.ndarray]:
    if rle_obj is None:
        return None
    rle = {"size": rle_obj["size"], "counts": rle_obj["counts"]}
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    m = coco_mask.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(bool)

def crop_with_context(base_im: Image.Image, bbox: List[float], pad: int = 32) -> Image.Image:
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(base_im.width, x2 + pad), min(base_im.height, y2 + pad)
    return base_im.crop((x1, y1, x2, y2))

def save_resized(im: Image.Image, path: str, max_side: int = 1280, quality: int = 92) -> None:
    w, h = im.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=quality)

def rle_signature(rle_obj: Optional[Dict[str, Any]]) -> str:
    if rle_obj is None:
        return "r:none"
    size = rle_obj.get("size", [])
    cnts = rle_obj.get("counts", "")
    if isinstance(cnts, str):
        cnts = cnts.encode("utf-8")
    raw = b"rle|" + str(size).encode("utf-8") + b"|" + cnts
    return hashlib.sha1(raw).hexdigest()[:16]

def bbox_signature(bbox: Optional[List[float]]) -> str:
    if bbox is None:
        return "b:none"
    vals = ",".join(f"{v:.2f}" for v in bbox)
    raw = f"bbox|{vals}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]

def overlay_signature(image_abs_path: str, ex_id: str,
                      regions: List[Tuple[Optional[Dict[str, Any]], Optional[List[float]]]]) -> str:
    try:
        st = Path(image_abs_path).stat()
        img_sig = f"{st.st_size}-{int(st.st_mtime)}"
    except Exception:
        img_sig = "nosize-nomtime"
    parts = [img_sig, ex_id or "noid"]
    for rle_i, bbox_i in regions:
        parts.append(rle_signature(rle_i))
        parts.append(bbox_signature(bbox_i))
    sig = "|".join(parts).encode("utf-8")
    return hashlib.sha1(sig).hexdigest()[:20]

def crop_signature(ex_id: str, rle_i: Optional[Dict[str, Any]], bbox_i: Optional[List[float]]) -> str:
    raw = f"{ex_id}|{rle_signature(rle_i)}|{bbox_signature(bbox_i)}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]

def cache_paths(img_path: str, ex_id: str,
                rles_and_bboxes: List[Tuple[Optional[Dict[str, Any]], Optional[List[float]]]],
                cache_dir: str) -> Tuple[str, List[str]]:
    stem = Path(img_path).stem
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    ov_sig = overlay_signature(img_path, ex_id, rles_and_bboxes)
    overlay = Path(cache_dir) / f"{stem}_ov_{ov_sig}.jpg"
    crop_paths = []
    for (rle_i, bbox_i) in rles_and_bboxes:
        csig = crop_signature(ex_id, rle_i, bbox_i)
        crop_paths.append(str(Path(cache_dir) / f"{stem}_crop_{csig}.jpg"))
    return str(overlay), crop_paths

def make_multi_region_overlay(full_img_path: str,
                              rles_and_bboxes: List[Tuple[Optional[Dict[str, Any]], Optional[List[float]]]],
                              colors=COLORS) -> Image.Image:
    with Image.open(full_img_path) as _im:
        im = _im.convert("RGB")
    W, H = im.size
    arr = np.array(im).astype(np.float32)
    dim = (arr * 0.25).astype(np.uint8)
    out = dim.copy()

    any_mask = False
    for i, (rle_obj, bbox) in enumerate(rles_and_bboxes):
        color = colors[i % len(colors)]
        mask = None
        if rle_obj is not None:
            mask = rle_to_mask(rle_obj)
        elif bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            mask = np.zeros((H, W), dtype=bool)
            mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
        if mask is not None:
            any_mask = True
            out[mask] = ((0.6 * arr[mask] + 0.4 * np.array(color)).clip(0, 255).astype(np.uint8))
    if not any_mask:
        return im
    return Image.fromarray(out)

def last_human_index(conv: List[Dict[str, Any]]) -> int:
    for i in reversed(range(len(conv))):
        if conv[i].get("from") == "human":
            return i
    return -1

def region_indices_from_last_question(conv: List[Dict[str, Any]], max_regions: int = 3) -> List[int]:
    h = last_human_index(conv)
    if h < 0:
        return [0]
    txt = conv[h].get("value", "") or ""
    idxs = [int(m) for m in REGION_PAT_ALL.findall(txt)]
    idxs = [max(0, k-1) for k in idxs][:max_regions]
    return idxs or [0]

def build_messages_from_sample(ex: Dict[str, Any],
                               img_root: str,
                               cache_dir: Optional[str] = None,
                               rebuild_cache: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    """
    データ1件（object-centric）から推論用メッセージを構築。
    戻り値: (messages, gt_answer)
    """
    conv = ex["conversations"]
    img_path = os.path.join(img_root, ex["image"])
    if not os.path.isfile(img_path):
        raise FileNotFoundError(img_path)
    full_abs = str(Path(img_path).resolve())

    region_idxs = region_indices_from_last_question(conv, max_regions=len(COLORS))

    # RLE/BBOX 収集
    rles_and_bboxes = []
    for ridx in region_idxs:
        rle_i = None
        if isinstance(ex.get("rle"), list) and len(ex["rle"]) > ridx:
            rle_i = ex["rle"][ridx]
        bbox_i = None
        if isinstance(ex.get("bbox"), list) and len(ex["bbox"]) > ridx:
            bx = ex["bbox"][ridx]
            bbox_i = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]
        rles_and_bboxes.append((rle_i, bbox_i))

    # キャッシュ先
    tmp_dir = cache_dir or os.path.join(img_root, "_cache_resize")
    overlay_path, crop_paths = cache_paths(full_abs, ex.get("id", "noid"), rles_and_bboxes, tmp_dir)

    # 画像生成
    if (not os.path.isfile(overlay_path)) or rebuild_cache:
        ov = make_multi_region_overlay(full_abs, rles_and_bboxes, colors=COLORS)
        save_resized(ov, overlay_path, max_side=1280)

    crop_paths_final = []
    need_crop = any(b is not None for (_, b) in rles_and_bboxes)
    if need_crop:
        with Image.open(full_abs) as _base:
            base_im = _base.convert("RGB")
            for (rle_i, bbox_i), cp in zip(rles_and_bboxes, crop_paths):
                if bbox_i is None:
                    continue
                if (not os.path.isfile(cp)) or rebuild_cache:
                    cr = crop_with_context(base_im, bbox_i, pad=32)
                    save_resized(cr, cp, max_side=1024)
                crop_paths_final.append(cp)

    # GT回答（最後のhuman直後のgpt）
    h = last_human_index(conv)
    assert h >= 0 and h+1 < len(conv) and conv[h+1].get("from") == "gpt", "bad sample sequence"
    gt_answer = (conv[h+1].get("value") or "").strip()

    # メッセージ構築（学習時と同等）
    sys_msg = {
        "role": "system",
        "content": (
            "画像1: 全景。画像2: 複数のRegionを色分けしてハイライト。"
            "画像3以降: Region順のクロップ（存在する場合）。"
            "『Region [k]』は画像2の色分けに対応する対象を指す。"
            "与えられた会話のスタイルに従って回答せよ。"
        ),
    }
    messages = [sys_msg]
    for i, turn in enumerate(conv[:h+1]):
        if turn.get("from") == "human":
            txt = (turn.get("value") or "").replace("<image>", "").replace("<mask>", "（ハイライト領域）").strip()
            content = [{"type": "text", "text": txt}]
            if i == 0:
                imgs = [
                    {"type": "image", "image": f"file://{full_abs}"},
                    {"type": "image", "image": f"file://{overlay_path}"},
                ]
                for cp in crop_paths_final:
                    imgs.append({"type": "image", "image": f"file://{cp}"})
                # BBOX情報も追記
                for j, ridx in enumerate(region_idxs):
                    bbox_j = rles_and_bboxes[j][1]
                    if bbox_j is not None:
                        btxt = f"REGION[{ridx + 1}]_BBOX={int(bbox_j[0])},{int(bbox_j[1])},{int(bbox_j[2])},{int(bbox_j[3])}"
                        content.append({"type": "text", "text": btxt})
                content = imgs + content
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "assistant", "content": turn.get("value", "")})

    return messages, gt_answer

# ================ モデルのロード/解放 ================
def load_base_model_and_processor() -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    return model, processor

def apply_lora_adapter(model: Qwen2_5_VLForConditionalGeneration, adapter_dir: str
                       ) -> Qwen2_5_VLForConditionalGeneration:
    # LoRAを一時的に適用（mergeはしない→VRAM負荷を抑える）
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    return model

def free_model(model, processor):
    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================ 推論 ================
def run_generate(model, processor, messages: List[Dict[str, Any]]) -> str:
    from qwen_vl_utils import process_vision_info  # ローカルにある想定
    # テンプレート整形
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=images,
        videos=(videos if (videos is not None and len(videos) > 0) else None),
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors="pt",
    )
    # デバイスへ
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KW)
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return text

# ================ メイン ================
def main(sample_index: int = 0):
    # 1) サンプル読み込み
    data = json.load(open(ANN_PATH, "r"))
    assert 0 <= sample_index < len(data), f"sample_index out of range: 0..{len(data)-1}"
    ex = data[sample_index]

    # メッセージ構築
    cache_dir = CACHE_DIR or os.path.join(IMG_ROOT, "_cache_resize")
    messages, gt = build_messages_from_sample(ex, IMG_ROOT, cache_dir, REBUILD_CACHE)
    print(f"[Sample] id={ex.get('id')} image={ex.get('image')}")
    print(f"[Question] {(messages[-1]['content'][0]['text'] if isinstance(messages[-1]['content'], list) else 'N/A')}\n")

    # 2) Baseモデルで推論 → 解放
    print("=== Inference: Base model ===")
    base_model, base_proc = load_base_model_and_processor()
    base_text = run_generate(base_model, base_proc, messages)
    print(base_text.strip(), "\n")
    free_model(base_model, base_proc)

    # 3) LoRA適用で推論 → 解放
    print("=== Inference: LoRA-applied model ===")
    lora_base, lora_proc = load_base_model_and_processor()    # 新規ロード（VRAM共存を避ける）
    lora_model = apply_lora_adapter(lora_base, ADAPTER_DIR)
    lora_text = run_generate(lora_model, lora_proc, messages)
    print(lora_text.strip(), "\n")
    free_model(lora_model, lora_proc)

    # 4) GT との見比べ
    print("=== Ground Truth (from dataset) ===")
    print(gt.strip())

# 追加: id→index検索
def find_index_by_id(data, sample_id: str) -> Optional[int]:
    for i, ex in enumerate(data):
        if str(ex.get("id")) == str(sample_id):
            return i
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_index", type=int, default=0,
                        help="評価するサンプルのインデックス（0始まり）")
    parser.add_argument("--sample_id", type=str, default=None,
                        help="評価するサンプルのid（例: 6d548f10）")
    parser.add_argument("--list", type=int, default=0,
                        help="先頭N件の (index, id, image) を一覧表示して終了。例: --list 10")
    args = parser.parse_args()

    data = json.load(open(ANN_PATH, "r"))

    # 一覧だけ見たい場合
    if args.list > 0:
        for i, ex in enumerate(data[:args.list]):
            print(f"{i:4d}  id={ex.get('id')}  image={ex.get('image')}")
        raise SystemExit(0)

    # id優先 → なければ index
    if args.sample_id is not None:
        idx = find_index_by_id(data, args.sample_id)
        if idx is None:
            raise ValueError(f"sample_id '{args.sample_id}' が見つかりません。"
                             " --list で候補を確認してください。")
        sample_index = idx
    else:
        sample_index = args.sample_index

    main(sample_index=sample_index)
