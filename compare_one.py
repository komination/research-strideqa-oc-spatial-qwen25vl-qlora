import argparse
import json
import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

YES_TOKENS = {"yes", "true", "correct", "indeed", "yep", "affirmative"}
NO_TOKENS  = {"no", "false", "incorrect", "not", "nope", "negative"}
NUM_PAT = re.compile(r"[-+]?\d+(?:\.\d+)?")
REG_PAT = re.compile(r"[Rr]egion\D*(\d+)")

def load_sample(data_json: str, sample_id: str) -> Dict[str, Any]:
    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for ex in data:
        if ex.get("id") == sample_id:
            return ex
    raise ValueError(f"sample_id='{sample_id}' が見つかりません: {data_json}")

def last_human_and_gt(conv: List[Dict[str, str]]) -> Tuple[str, str]:
    last_h = None
    for i in reversed(range(len(conv))):
        if conv[i].get("from") == "human":
            last_h = i
            break
    if last_h is None or last_h + 1 >= len(conv) or conv[last_h + 1].get("from") != "gpt":
        raise ValueError("最後の human に続く gpt の正解が見つかりません")
    q = (conv[last_h]["value"] or "").replace("<image>", "").replace("<mask>", "（対象領域）").strip()
    a = (conv[last_h + 1]["value"] or "").strip()
    return q, a

def build_messages(image_path: str, question_text: str) -> List[Dict[str, Any]]:
    """
    Qwen のマルチモーダル chat 形式。
    ここでは system は空、user の最初に image、続けてテキスト。
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": question_text},
            ],
        }
    ]

def prepare_inputs(processor, messages: List[Dict[str, Any]], max_len: int = 4096):
    prompt_txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    kwargs = dict(
        text=[prompt_txt],
        images=imgs,
        videos=(vids if vids else None),
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors="pt",
    )
    return prompt_txt, processor(**kwargs)

# 生成部分だけを厳密に抽出（compare_one.py の generate を置き換え）
def generate(model, processor, messages, max_new_tokens=128, temperature=0.0, top_p=1.0) -> str:
    prompt_txt, enc = prepare_inputs(processor, messages)
    enc = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    gen_ids = out_ids[0, input_len:]
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def first_float(s: str) -> Optional[float]:
    m = NUM_PAT.search(s.replace(",", ""))
    return float(m.group(0)) if m else None

def normalize_str(s: str) -> str:
    s = s.lower().strip()
    tbl = str.maketrans("", "", string.punctuation)
    return s.translate(tbl).replace("  ", " ")

def yesno_from_text(s: str) -> Optional[bool]:
    t = normalize_str(s)
    has_yes = any(tok in t.split() for tok in YES_TOKENS)
    has_no  = any(tok in t.split() for tok in NO_TOKENS)
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None

def region_id_from_text(s: str) -> Optional[int]:
    m = REG_PAT.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def simple_eval(question: str, gt: str, pred: str) -> Dict[str, Any]:
    # 1) Region 選択（最優先）
    def region_id_from_text(s: str) -> Optional[int]:
        m = REG_PAT.search(s)
        return int(m.group(1)) if m else None

    gt_reg = region_id_from_text(gt)
    if gt_reg is not None:
        pr_reg = region_id_from_text(pred)
        ok = (pr_reg is not None and pr_reg == gt_reg)
        return {"type": "region", "ok": ok, "gt": gt_reg, "pred": pr_reg}

    # 2) Yes/No
    gt_bool = yesno_from_text(gt)
    if gt_bool is not None:
        pr_bool = yesno_from_text(pred)
        ok = (pr_bool is not None and pr_bool == gt_bool)
        return {"type": "boolean", "ok": ok, "gt": gt_bool, "pred": pr_bool}

    # 3) 数値（最後に回す）
    gt_num = first_float(gt)
    if gt_num is not None:
        pr_num = first_float(pred)
        if pr_num is None:
            return {"type": "numeric", "ok": False, "gt": gt_num, "pred": None, "rel_err": None}
        rel = abs(pr_num - gt_num) / max(abs(gt_num), 1e-6)
        ok = rel <= 0.25
        return {"type": "numeric", "ok": ok, "gt": gt_num, "pred": pr_num, "rel_err": rel}

    ok = (normalize_str(gt) == normalize_str(pred))
    return {"type": "string", "ok": ok, "gt": gt, "pred": pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--sample_id", required=True)
    ap.add_argument("--lora_path", required=True, help="学習後の LoRA アダプタ保存先 (…/lora_adapter)")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    # サンプル読み出し
    ex = load_sample(args.data_json, args.sample_id)
    img_path = os.path.join(args.img_root, ex["image"])
    question, gt_answer = last_human_and_gt(ex["conversations"])

    # モデル & プロセッサ読込
    bnb = None
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    def load_base():
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        min_pixels=256*28*28,
        max_pixels=512*28*28,
    )

    # Base
    base = load_base()
    base.eval()

    # QLoRA反映（PEFTをアタッチ）
    ft = load_base()
    ft = PeftModel.from_pretrained(ft, args.lora_path)
    ft.eval()

    # メッセージ構築（全景＋最後のhuman質問）
    msgs = build_messages(img_path, question)

    # 生成（温度0で再現性重視）
    base_pred = generate(base, processor, msgs, max_new_tokens=args.max_new_tokens, temperature=0.0)
    ft_pred   = generate(ft,   processor, msgs, max_new_tokens=args.max_new_tokens, temperature=0.0)

    # 簡易評価
    base_eval = simple_eval(question, gt_answer, base_pred)
    ft_eval   = simple_eval(question, gt_answer, ft_pred)

    print("\n===== Sample =====")
    print("ID:         ", args.sample_id)
    print("Image:      ", img_path)
    print("Question:   ", question)
    print("GT Answer:  ", gt_answer)
    print("\n--- Base ---")
    print(base_pred)
    print("Eval:", base_eval)
    print("\n--- QLoRA ---")
    print(ft_pred)
    print("Eval:", ft_eval)

    def to_score(d):
        if d["type"] == "numeric":
            return (1 if d["ok"] else 0, f"rel_err={d['rel_err']:.3f}" if d["rel_err"] is not None else "N/A")
        return (1 if d["ok"] else 0, "")
    b_s, b_m = to_score(base_eval)
    f_s, f_m = to_score(ft_eval)
    print("\n===== Summary =====")
    print(f"Base : {'OK' if b_s else 'NG'} {b_m}")
    print(f"QLoRA: {'OK' if f_s else 'NG'} {f_m}")

if __name__ == "__main__":
    main()
