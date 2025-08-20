import os
import re
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from pycocotools import mask as coco_mask

IMG_ROOT = "data/stride-qa-mini/images"
ANN_PATH = "data/stride-qa-mini/object_centric_spatial_qa.json"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
OUT_DIR = "qwen25vl_3b_obj_center_qlora"

# 画像リサイズとトークン長
MAX_SIDE_FULL = 896     # 全景/オーバレイ最大辺
MAX_SIDE_CROP = 768     # クロップ最大辺
MAX_TEXT_LEN = 4096     # 入力テキスト最大トークン
MAX_EXTRA_CROPS = 2     # 1サンプルのクロップ最大枚数

# オーバレイに Region ラベルを描くか
DRAW_REGION_LABELS = True
# 最初の user ターンに BBOX テキストメタを付与するか
WITH_BBOX_TEXT = True

# キャッシュ
CACHE_DIR = None
REBUILD_CACHE = False

# デバッグ用
MAX_SAMPLES = 0  # 0=無制限
SANITY_CHECK = False  # True で1サンプルだけ前処理→forwardを試す

# 複数 Region 用の色（全景ハイライト合成用）
COLORS = [
    (0, 180, 255),   # R1: 青
    (255, 80, 80),   # R2: 赤
    (120, 220, 120), # R3: 緑
    (255, 200, 0),   # R4: 黄
]

# ====================== 画像ユーティリティ ======================
def rle_to_mask(rle_obj: Dict[str, Any]) -> Optional[np.ndarray]:
    """COCO RLEをbool maskに。countsをbytes化し、(H,W,1)→(H,W)に落とす。"""
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


# ---------- シグネチャ（キャッシュ鍵） ----------
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


def overlay_signature(image_abs_path: str,
                      ex_id: str,
                      regions: List[Tuple[Optional[Dict[str, Any]], Optional[List[float]]]]
                      ) -> str:
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


def cache_paths(
    img_path: str,
    ex_id: str,
    rles_and_bboxes: List[Tuple[Optional[Dict[str, Any]], Optional[List[float]]]],
    cache_dir: str
) -> Tuple[str, List[str]]:
    """
    キャッシュの一意化：
      - overlay: 画像サイズ/mtime + ex_id + 各RegionのRLE/BBOXシグネチャ
      - crop:    ex_id + 該当RegionのRLE/BBOXシグネチャ
    """
    stem = Path(img_path).stem
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    ov_sig = overlay_signature(img_path, ex_id, rles_and_bboxes)
    overlay = Path(cache_dir) / f"{stem}_ov_{ov_sig}.jpg"

    crop_paths = []
    for (rle_i, bbox_i) in rles_and_bboxes:
        csig = crop_signature(ex_id, rle_i, bbox_i)
        crop_paths.append(str(Path(cache_dir) / f"{stem}_crop_{csig}.jpg"))

    return str(overlay), crop_paths


# ====================== ハイライト生成 ======================
def _label_anchor(
    W: int,
    H: int,
    rle_obj: Optional[Dict[str, Any]],
    bbox: Optional[List[float]],
) -> Tuple[int, int]:
    """ラベル描画位置（左上基準）を返す。bbox優先、無ければrleの重心近傍。"""
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        x = max(0, min(W - 1, x1 + 4))
        y = max(0, min(H - 1, y1 + 4))
        return x, y
    if rle_obj is not None:
        m = rle_to_mask(rle_obj)
        if m is not None and m.any():
            ys, xs = np.where(m)
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            x = max(0, min(W - 1, cx))
            y = max(0, min(H - 1, cy))
            return x, y
    return 8, 8


def make_multi_region_overlay(
    full_img_path: str,
    rles_and_bboxes: List[Tuple[Optional[Dict[str, Any]], Optional[List[float]]]],
    colors: List[Tuple[int, int, int]] = COLORS,
    region_ids: Optional[List[int]] = None,
    draw_labels: bool = True,
) -> Image.Image:
    """複数 Region を色分けし、対象以外を減光したハイライト画像を生成。必要なら R{k} を描画。"""
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
            mask[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)] = True

        if mask is not None:
            any_mask = True
            out[mask] = (
                (0.6 * arr[mask] + 0.4 * np.array(color)).clip(0, 255).astype(np.uint8)
            )

    if not any_mask:
        return Image.fromarray(arr.astype(np.uint8))

    overlay = Image.fromarray(out)

    # ラベル描画（R{k}）
    if draw_labels and region_ids is not None:
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for i, (rle_obj, bbox) in enumerate(rles_and_bboxes):
            rid = region_ids[i] if i < len(region_ids) else (i + 1)
            x, y = _label_anchor(W, H, rle_obj, bbox)
            label = f"R{rid}"
            # 白縁取り（視認性）
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                draw.text((x+dx, y+dy), label, fill=(0,0,0), font=font)
            draw.text((x, y), label, fill=(255,255,255), font=font)

    return overlay


# ====================== Region番号の抽出/対応付け ======================
REGION_PAT_ALL = re.compile(r"Region\s*\[(\d+)\]")

def find_region_ids_from_last_question(conv: List[Dict[str, Any]], max_regions: int = 3) -> List[int]:
    """
    会話文の「最後のhuman（学習対象の直前）」から 'Region [k]' をすべて抽出。
    ここで抽出するのは 0 始まりの「RegionID」そのもの（indexではない）。
    """
    last_h = None
    for i in reversed(range(len(conv))):
        if conv[i].get("from") == "human":
            last_h = i
            break
    if last_h is None:
        return [0]
    txt = conv[last_h].get("value", "") or ""
    ids = [int(m) for m in REGION_PAT_ALL.findall(txt)]
    return ids[:max_regions] or [0]


def build_region_id_to_pos(ex: Dict[str, Any]) -> Dict[int, int]:
    """
    ex['region'] が [[2,0]] のように入っているケースを正規化して {2:0, 0:1} を返す。
    """
    reg = ex.get("region", [])
    if isinstance(reg, list) and len(reg) > 0 and isinstance(reg[0], list):
        reg = reg[0]
    id2pos: Dict[int, int] = {}
    for pos, rid in enumerate(reg):
        try:
            id2pos[int(rid)] = pos
        except Exception:
            continue
    return id2pos


# ====================== Dataset ======================
class StrideQADataset(Dataset):
    """
    最後のhumanから参照されるRegion群を抽出し、
    - 画像1: 全景
    - 画像2: ハイライト（複数Regionを色分け、必要なら R{k} ラベル）
    - 画像3以降: Region順に各クロップ（上限 MAX_EXTRA_CROPS）
    を user の最初のターンに与える。以降はテキストのみ。
    """
    def __init__(self, path: str, img_root: str):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r") as f:
            raw = json.load(f)

        print(f"[Dataset] preprocessing {len(raw)} samples ...", flush=True)

        def is_valid_conv(conv):
            return (
                isinstance(conv, list)
                and any(t.get("from") == "human" for t in conv)
                and any(t.get("from") == "gpt" for t in conv)
            )

        for idx, ex in enumerate(raw):
            if "conversations" not in ex or "image" not in ex:
                continue
            conv = ex["conversations"]
            if not is_valid_conv(conv):
                continue

            img_path = os.path.join(img_root, ex["image"])
            if not os.path.isfile(img_path):
                continue

            # --- RegionID 群を抽出 ---
            region_ids = find_region_ids_from_last_question(conv, max_regions=len(COLORS))
            id2pos = build_region_id_to_pos(ex)

            # --- RLE / BBOX を抽出（RegionID → 配列位置へ写像） ---
            rles_and_bboxes: List[Tuple[Optional[Dict[str, Any]], Optional[List[float]]]] = []
            for rid in region_ids:
                pos = id2pos.get(rid, None)
                rle_i = None
                if pos is not None and isinstance(ex.get("rle"), list) and len(ex["rle"]) > pos:
                    rle_i = ex["rle"][pos]

                bbox_i = None
                if pos is not None and isinstance(ex.get("bbox"), list) and len(ex["bbox"]) > pos:
                    bx = ex["bbox"][pos]
                    # 想定: [x1,y1,x2,y2]（もし [x,y,w,h] の場合は必要に応じて変換を入れる）
                    if bx is not None and len(bx) >= 4:
                        bbox_i = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]

                rles_and_bboxes.append((rle_i, bbox_i))

            # --- 1つもマスク/ボックスが無ければスキップ ---
            if all((r is None and b is None) for (r, b) in rles_and_bboxes):
                continue

            # --- キャッシュ先決定（シグネチャを使って衝突回避） ---
            full_abs = str(Path(img_path).resolve())
            tmp_dir = CACHE_DIR or os.path.join(img_root, "_cache_resize")
            overlay_path, crop_paths = cache_paths(full_abs, ex.get("id", "noid"), rles_and_bboxes, tmp_dir)

            # 既存確認
            have_overlay = os.path.isfile(overlay_path)
            have_crops = [os.path.isfile(cp) for cp in crop_paths]
            if REBUILD_CACHE:
                have_overlay = False
                have_crops = [False] * len(crop_paths)

            # --- 画像生成（必要なもののみ） ---
            if not have_overlay:
                overlay_im = make_multi_region_overlay(
                    full_abs,
                    rles_and_bboxes,
                    colors=COLORS,
                    region_ids=region_ids,
                    draw_labels=DRAW_REGION_LABELS,
                )
                save_resized(overlay_im, overlay_path, max_side=MAX_SIDE_FULL, quality=92)

            crop_paths_final: List[str] = []
            need_crop = any(b is not None for (_, b) in rles_and_bboxes)
            if need_crop:
                with Image.open(full_abs) as _base:
                    base_im = _base.convert("RGB")
                    produced = 0
                    for (rle_i, bbox_i), cp, ok in zip(rles_and_bboxes, crop_paths, have_crops):
                        if bbox_i is None:
                            continue
                        if produced >= MAX_EXTRA_CROPS:
                            break
                        if not ok:
                            crop_im = crop_with_context(base_im, bbox_i, pad=32)
                            save_resized(crop_im, cp, max_side=MAX_SIDE_CROP, quality=92)
                        crop_paths_final.append(cp)
                        produced += 1

            # --- 学習用アシスタントの答え（最後のhumanの直後gpt） ---
            try:
                last_h = max(i for i, t in enumerate(conv) if t.get("from") == "human")
            except ValueError:
                continue
            if last_h + 1 >= len(conv) or conv[last_h + 1].get("from") != "gpt":
                continue
            answer = (conv[last_h + 1].get("value") or "").strip()
            if not answer:
                continue

            # --- メッセージ構築 ---
            sys_msg = {
                "role": "system",
                "content": (
                    "画像1: 全景。画像2: 複数のRegionを色分けしてハイライト（R{k} ラベル）。"
                    "画像3以降: Region順のクロップ（存在する場合、枚数は上限あり）。"
                    "『Region [k]』は画像2の R{k} に対応する対象を指す。"
                    "回答スタイルは与えられた会話例に従うこと。"
                ),
            }
            messages = [sys_msg]

            for i, turn in enumerate(conv[: last_h + 1]):
                if turn.get("from") == "human":
                    txt = (turn.get("value") or "").replace("<image>", "").replace("<mask>", "（ハイライト領域）").strip()

                    content: List[Dict[str, Any]] = [{"type": "text", "text": txt}]
                    if i == 0:
                        # 最初のhumanターンに画像を前置き
                        imgs = [
                            {"type": "image", "image": full_abs},       # file:// は使わない
                            {"type": "image", "image": overlay_path},
                        ]
                        for cp in crop_paths_final:
                            imgs.append({"type": "image", "image": cp})

                        # BBOX 情報を追記（任意・自然文寄りに）
                        if WITH_BBOX_TEXT:
                            for j, rid in enumerate(region_ids):
                                bbox_j = rles_and_bboxes[j][1] if j < len(rles_and_bboxes) else None
                                if bbox_j is not None:
                                    btxt = f"Region [{rid}] bbox: ({int(bbox_j[0])},{int(bbox_j[1])})-({int(bbox_j[2])},{int(bbox_j[3])})"
                                    content.append({"type": "text", "text": btxt})

                        content = imgs + content

                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "assistant", "content": turn.get("value", "")})

            self.items.append({"messages": messages, "answer": answer})

            # 進捗ログ
            if (idx + 1) % 100 == 0:
                print(f"[Dataset] processed {idx + 1}/{len(raw)} (valid={len(self.items)})", flush=True)

            if MAX_SAMPLES and len(self.items) >= MAX_SAMPLES:
                print(f"[Dataset] early stop at {len(self.items)} samples (MAX_SAMPLES)", flush=True)
                break

        if not self.items:
            raise RuntimeError(f"No valid samples after preprocessing: {path}")
        print(f"[Dataset] region-aware samples: {len(self.items)}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ====================== Collator ======================
@dataclass
class QwenCollator:
    processor: Any

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        img_lists, vid_lists = [], []
        prompt_lens: List[int] = []

        for i, ex in enumerate(batch):
            if not isinstance(ex, dict) or "messages" not in ex or "answer" not in ex:
                raise ValueError(f"Bad sample at index {i}: {ex}")
            msgs = ex["messages"]

            # chat template（生成プロンプト付与）→ 画像・動画抽出
            prompt_txt = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            img_in, vid_in = process_vision_info(msgs)

            # プロンプト長は本番と同じ設定で測る
            pin = self.processor(
                text=[prompt_txt],
                images=img_in,
                videos=(vid_in if (vid_in is not None and len(vid_in) > 0) else None),
                padding=False,
                truncation=True,
                max_length=MAX_TEXT_LEN,
                return_tensors="pt",
            )
            prompt_lens.append(pin["input_ids"].shape[1])

            # 教師：prompt + answer
            texts.append(prompt_txt + ex["answer"])
            img_lists.append(img_in)
            vid_lists.append(vid_in)

        has_any_video = any(v is not None and len(v) > 0 for v in vid_lists)
        proc_kwargs = dict(
            text=texts,
            images=img_lists,
            padding="longest",
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        if has_any_video:
            proc_kwargs["videos"] = vid_lists

        fin = self.processor(**proc_kwargs)
        input_ids = fin["input_ids"]
        labels = input_ids.clone()

        # プロンプト部を無効化（長さはシーケンス長でクリップ）
        seq_len = labels.shape[1]
        for i, plen in enumerate(prompt_lens):
            plen = min(plen, seq_len)
            labels[i, :plen] = -100

        # PADも無効化
        if "attention_mask" in fin:
            labels[fin["attention_mask"] == 0] = -100

        out = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": fin.get("attention_mask"),
        }
        # 画像テンソル類
        for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            if k in fin:
                out[k] = fin[k]
        return out


# ====================== モデル/プロセッサ ======================
def guess_lora_targets_minimal(model: nn.Module) -> List[str]:
    """
    LoRA ターゲット：LM（q/k/v/o, gate/up/down）＋ mm_projector 系の線形層に限定。
    vision_tower は初手では凍結しておく。
    """
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    lm_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    proj_keys = [
        "mm_projector",
        "multi_modal_projector",
        "visual_projector",
        "mm_resampler",            # 実装差吸収（存在しない場合も多い）
        "projector",               # 保険（過マッチし過ぎないよう注意）
    ]
    lm_targets = [n for n in linear_names if any(k in n for k in lm_keys)]
    proj_targets = [n for n in linear_names if any(k in n for k in proj_keys)]
    targets = sorted(set(lm_targets + proj_targets))
    print("[LoRA targets] total count =", len(targets))
    for t in targets[:40]:
        print("  -", t)
    missing_proj = [k for k in proj_keys if not any(k in n for n in targets)]
    if missing_proj:
        print("[WARN] projector-like modules not found (may be fine):", missing_proj)
    if not targets:
        # フォールバック
        targets = lm_targets or linear_names[:32]
        print("[LoRA targets] fallback used. count =", len(targets))
    return targets


def get_model_and_processor():
    # 可能なら TF32 を許可（Ampere+）
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # try:
    #     model.config.attn_implementation = "flash_attention_2"
    # except Exception:
    #     pass

    # QLoRA 前処理
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA: LM + projector に限定
    targets = guess_lora_targets_minimal(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Processor（OOM回避のためmax_pixelsは抑えめ）
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=384 * 28 * 28,
    )
    return model, processor


# ====================== Sanity Check（任意） ======================
def sanity_check_one(train_set: Dataset, model: nn.Module, processor: Any):
    print("[Sanity] running single-sample encode/forward ...")
    ex = train_set[0]
    msgs = ex["messages"]
    ans = ex["answer"]
    txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    pin = processor(text=[txt], images=imgs,
                    videos=(vids if vids else None),
                    truncation=True, max_length=MAX_TEXT_LEN, return_tensors="pt")
    fin = processor(text=[txt + ans], images=imgs,
                    videos=(vids if vids else None),
                    truncation=True, max_length=MAX_TEXT_LEN, return_tensors="pt")
    batch = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in fin.items()}
    with torch.no_grad():
        _ = model(**batch)
    print("[Sanity] OK. seq_len:", fin["input_ids"].shape[1])


# ====================== メイン ======================
def main():
    # データセット
    train_set = StrideQADataset(ANN_PATH, IMG_ROOT)

    # モデル/プロセッサ
    model, processor = get_model_and_processor()

    # Sanity（任意）
    if SANITY_CHECK:
        sanity_check_one(train_set, model, processor)

    # Collator
    collator = QwenCollator(processor)

    # 学習設定（まずは安全側。必要なら lr を 1e-4 に上げる）
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=0.3,
        seed=42,
        group_by_length=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        data_collator=collator,
    )

    trainer.train()

    # LoRAアダプタのみ保存
    save_dir = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(save_dir)
    print("[DONE] Saved LoRA adapter to", save_dir)


if __name__ == "__main__":
    main()
