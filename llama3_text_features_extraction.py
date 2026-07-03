#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama3 part-text feature extraction for Talk2DINO Stage2 part training.

This script follows the original UVLT extraction logic:
    part name -> chat dialog -> Llama.chat_completion_feat()
    -> generated assistant response corr_feats -> token pooling -> 4096-D feature.

Default behavior when run from Talk2DINO root:
    1) Load local Llama3 from ./Meta-Llama-3-8B-Instruct
    2) Read prompts from ./llama3_part_prompts.txt
    3) Infer VOC116 part names from test15 train/val pth by part_category_id / part_class_name
    4) Extract a class-level Llama3 part bank [116, num_prompts*num_epochs, 4096]
    5) Add ann['llama_part_ann_feats'] into train/val pth

Object feature ann['ann_feats'] is kept unchanged.
Original CLIP part feature ann['part_ann_feats'] is kept unchanged unless --replace_part_ann_feats is set.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llama import Llama


# -----------------------------
# Defaults for current Talk2DINO layout
# -----------------------------

DEFAULT_CKPT_DIR = "Meta-Llama-3-8B-Instruct"
DEFAULT_TOKENIZER_PATH = "Meta-Llama-3-8B-Instruct/tokenizer.model"
DEFAULT_PROMPT_FILE = "llama3_part_prompts.txt"
DEFAULT_BANK_OUT = "feature_new_exp/voc116/part/llama3_part_stage2_v1.pt"
DEFAULT_INPUT_PTHS = [
    "feature/voc116_obj_part_test15/train_voc116_obj_with_text.pth",
    "feature/voc116_obj_part_test15/val_voc116_obj_with_text.pth",
]
DEFAULT_OUTPUT_PTHS = [
    "feature/voc116_obj_part_test15/train_voc116_obj_with_llama3_part.pth",
    "feature/voc116_obj_part_test15/val_voc116_obj_with_llama3_part.pth",
]

DEFAULT_SYSTEM_TEMPLATE = (
    "You are a visual part description assistant. "
    "Answer only with concise visual appearance information useful for image segmentation. "
    "Avoid function, usage, behavior, and non-visual commonsense."
)


# -----------------------------
# Basic utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_env(master_port: str) -> None:
    """The Meta Llama reference code expects a distributed process group."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")


def chunked(xs: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def to_int_list(x: Any) -> List[int]:
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().view(-1).tolist()]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def to_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]


# -----------------------------
# Prompt loading
# -----------------------------

def load_prompt_file(path: str) -> Tuple[List[str], str]:
    """
    Supported formats:
      txt: one prompt per non-empty line, # as comment
      json list: ["prompt 1", "prompt 2"]
      json dict: {"prompts": [...], "system": "optional system template"}

    Prompt templates may use {name}, {cls}, or [cls].
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    if path_obj.suffix.lower() == ".json":
        obj = json.loads(path_obj.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            prompts = [str(x).strip() for x in obj if str(x).strip()]
            system = ""
        elif isinstance(obj, dict):
            prompts = [str(x).strip() for x in obj["prompts"] if str(x).strip()]
            system = str(obj.get("system", "")).strip()
        else:
            raise ValueError("JSON prompt file must be a list or a dict with key 'prompts'.")
    else:
        prompts = []
        for line in path_obj.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
        system = ""

    if len(prompts) == 0:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompts, system


def fill_prompt(template: str, name: str, bracket_cls: bool) -> str:
    cls_text = f"[{name}]" if bracket_cls else name
    out = template.replace("[cls]", cls_text)
    out = out.replace("{name}", name)
    out = out.replace("{cls}", cls_text)
    return out


def build_dialog(system_template: str, prompt_template: str, name: str, bracket_cls: bool) -> List[Dict[str, str]]:
    user_prompt = fill_prompt(prompt_template, name, bracket_cls=bracket_cls)
    system_content = system_template.replace("[prompt]", prompt_template).replace("{prompt}", prompt_template)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]


# -----------------------------
# Infer part names from Talk2DINO pth
# -----------------------------

def infer_part_names_from_pths(
    pth_paths: Sequence[str],
    part_id_key: str,
    part_name_key: str,
    expected_num_parts: int,
) -> List[str]:
    """
    Build names[id] from annotation fields part_category_id and part_class_name.
    This avoids importing uvlt/dataloaders in Talk2DINO.
    """
    id_to_name: Dict[int, str] = {}

    for pth_path in pth_paths:
        data = torch.load(pth_path, map_location="cpu")
        if "annotations" not in data:
            raise KeyError(f"{pth_path} has no top-level key 'annotations'.")

        for ann_idx, ann in enumerate(data["annotations"]):
            if part_id_key not in ann:
                continue
            if part_name_key not in ann:
                raise KeyError(
                    f"Annotation {ann_idx} in {pth_path} has '{part_id_key}' but no '{part_name_key}'. "
                    f"Available keys: {list(ann.keys())}"
                )

            part_ids = to_int_list(ann[part_id_key])
            part_names = to_str_list(ann[part_name_key])
            if len(part_ids) != len(part_names):
                raise ValueError(
                    f"Annotation {ann_idx} in {pth_path}: len({part_id_key})={len(part_ids)} "
                    f"but len({part_name_key})={len(part_names)}."
                )

            for pid, pname in zip(part_ids, part_names):
                if pid in id_to_name and id_to_name[pid] != pname:
                    raise ValueError(
                        f"Conflicting name for part id {pid}: '{id_to_name[pid]}' vs '{pname}' "
                        f"in {pth_path}, annotation {ann_idx}."
                    )
                id_to_name[pid] = pname

    if len(id_to_name) == 0:
        raise RuntimeError(
            f"Could not infer any part names from {pth_paths}. "
            f"Need annotation keys '{part_id_key}' and '{part_name_key}'."
        )

    max_id = max(id_to_name.keys())
    names: List[str] = []
    missing: List[int] = []
    for pid in range(max_id + 1):
        if pid not in id_to_name:
            missing.append(pid)
            names.append(f"__missing_part_{pid}__")
        else:
            names.append(id_to_name[pid])

    if missing:
        raise RuntimeError(
            f"Missing part names for ids: {missing[:30]}{'...' if len(missing) > 30 else ''}. "
            "Do not continue because feature-bank row ids would be ambiguous."
        )

    if expected_num_parts > 0 and len(names) != expected_num_parts:
        raise RuntimeError(
            f"Expected {expected_num_parts} parts, but inferred {len(names)} parts from pth. "
            f"max_id={max_id}."
        )

    return names


# -----------------------------
# Llama feature extraction
# -----------------------------

@torch.inference_mode()
def run_chat_completion_feat(
    generator: Llama,
    dialogs: List[List[Dict[str, str]]],
    max_gen_len: int,
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    return generator.chat_completion_feat(
        dialogs,  # type: ignore[arg-type]
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )


def pool_corr_feats(corr_feats: List[torch.Tensor], pool: str) -> torch.Tensor:
    if len(corr_feats) == 0:
        raise RuntimeError("chat_completion_feat returned empty corr_feats; increase max_gen_len or check stop tokens.")
    feats = torch.stack([x.detach().float().cpu() for x in corr_feats], dim=0)  # [T, 4096]
    if pool == "mean":
        return feats.mean(dim=0)
    if pool == "last":
        return feats[-1]
    raise ValueError(f"Unknown response_pool: {pool}")


def extract_llama_part_bank(args: argparse.Namespace, names: List[str]) -> Dict[str, Any]:
    prompts, system_from_file = load_prompt_file(args.prompt_file)
    system_template = system_from_file if (system_from_file and not args.ignore_system_in_prompt_file) else args.system_template

    if args.limit_classes > 0:
        names = names[:args.limit_classes]

    init_distributed_env(args.master_port)
    set_seed(args.seed)

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
        local_rank=args.local_rank,
        MASTER_PORT=str(args.master_port),
    )

    all_feats_by_epoch_prompt: List[torch.Tensor] = []
    generated_texts: List[List[List[str]]] = []  # [epoch][class][prompt]

    for epoch in range(args.num_epochs):
        epoch_prompt_feats: List[torch.Tensor] = []
        epoch_generated_texts: List[List[str]] = [[] for _ in names]

        for prompt_idx, prompt_template in enumerate(prompts):
            dialogs = [
                build_dialog(system_template, prompt_template, name, bracket_cls=args.bracket_cls)
                for name in names
            ]

            prompt_feats: List[torch.Tensor] = []
            batches = list(chunked(list(enumerate(dialogs)), args.max_batch_size))
            pbar = tqdm(batches, desc=f"epoch {epoch + 1}/{args.num_epochs}, prompt {prompt_idx + 1}/{len(prompts)}")

            for batch in pbar:
                batch_indices = [idx for idx, _ in batch]
                batch_dialogs = [dialog for _, dialog in batch]
                outputs = run_chat_completion_feat(
                    generator=generator,
                    dialogs=batch_dialogs,
                    max_gen_len=args.max_gen_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                for local_i, out in enumerate(outputs):
                    gen = out["generation"]
                    corr_feats = gen["corr_feats"]
                    feat = pool_corr_feats(corr_feats, pool=args.response_pool)
                    if args.normalize:
                        feat = F.normalize(feat.float(), dim=0)
                    prompt_feats.append(feat.cpu())
                    content = str(gen.get("content", ""))
                    epoch_generated_texts[batch_indices[local_i]].append(content)

            prompt_feat_tensor = torch.stack(prompt_feats, dim=0)  # [C, 4096]
            epoch_prompt_feats.append(prompt_feat_tensor)

        # [P, C, D] -> [C, P, D]
        epoch_feat_tensor = torch.stack(epoch_prompt_feats, dim=0).permute(1, 0, 2).contiguous()
        all_feats_by_epoch_prompt.append(epoch_feat_tensor)
        generated_texts.append(epoch_generated_texts)

    # [E, C, P, D] -> [C, E*P, D]
    feats = torch.stack(all_feats_by_epoch_prompt, dim=0).permute(1, 0, 2, 3).contiguous()
    num_classes, num_epochs, num_prompts, dim = feats.shape
    feats = feats.view(num_classes, num_epochs * num_prompts, dim)

    mean_feats = feats.float().mean(dim=1)
    if args.normalize:
        mean_feats = F.normalize(mean_feats, dim=-1)

    return {
        "feats": feats.half() if args.save_half else feats.float(),
        "mean_feats": mean_feats.half() if args.save_half else mean_feats.float(),
        "names": names,
        "prompts": prompts,
        "system_template": system_template,
        "generated_texts": generated_texts,
        "meta": {
            "logic": "generated_response_corr_feats_pooling",
            "response_pool": args.response_pool,
            "num_epochs": args.num_epochs,
            "num_prompts": len(prompts),
            "dim": dim,
            "normalize": args.normalize,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_gen_len": args.max_gen_len,
            "max_seq_len": args.max_seq_len,
            "bracket_cls": args.bracket_cls,
            "part_id_key": args.part_id_key,
            "part_name_key": args.part_name_key,
        },
    }


# -----------------------------
# Inject into Talk2DINO pth
# -----------------------------

def inject_part_features_into_pth(
    input_pth: str,
    output_pth: str,
    llama_part_feats: torch.Tensor,
    part_id_key: str,
    output_part_key: str,
    replace_part_ann_feats: bool,
    dtype: str,
) -> None:
    data = torch.load(input_pth, map_location="cpu")
    if "annotations" not in data:
        raise KeyError(f"{input_pth} has no top-level key 'annotations'.")

    feats = llama_part_feats.detach().cpu().float()
    out_dtype = torch.float16 if dtype == "float16" else torch.float32

    for ann_idx, ann in enumerate(tqdm(data["annotations"], desc=f"inject {Path(input_pth).name}")):
        if part_id_key not in ann:
            raise KeyError(
                f"Annotation {ann_idx} does not contain key '{part_id_key}'. "
                f"Available keys: {list(ann.keys())}"
            )
        part_ids = to_int_list(ann[part_id_key])
        if len(part_ids) == 0:
            part_tensor = torch.zeros((0, feats.shape[-1]), dtype=out_dtype)
        else:
            min_id, max_id = min(part_ids), max(part_ids)
            if min_id < 0 or max_id >= feats.shape[0]:
                raise IndexError(
                    f"Annotation {ann_idx} has part id outside feature bank range: "
                    f"min={min_id}, max={max_id}, bank_size={feats.shape[0]}"
                )
            part_idx = torch.as_tensor(part_ids, dtype=torch.long, device=feats.device)
            part_tensor = feats.index_select(0, part_idx).to(dtype=out_dtype).cpu()

        ann[output_part_key] = part_tensor
        if replace_part_ann_feats:
            ann["part_ann_feats"] = part_tensor

    out_dir = os.path.dirname(output_pth)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, output_pth)
    print(f"[OK] wrote {output_pth}")
    print(f"     added field: {output_part_key}, dim={feats.shape[-1]}, dtype={out_dtype}")
    if replace_part_ann_feats:
        print("     also replaced original field: part_ann_feats")


def parse_io_pairs(input_pths: List[str], output_pths: List[str]) -> List[Tuple[str, str]]:
    if len(input_pths) != len(output_pths):
        raise ValueError("--input_pth and --output_pth must have the same number of values.")
    return list(zip(input_pths, output_pths))


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Llama3 generated-response features for VOC116 parts and inject into Talk2DINO test15 pth."
    )

    # Defaults match current Talk2DINO root layout.
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--bank_out", type=str, default=DEFAULT_BANK_OUT)
    parser.add_argument("--input_pth", type=str, nargs="*", default=DEFAULT_INPUT_PTHS)
    parser.add_argument("--output_pth", type=str, nargs="*", default=DEFAULT_OUTPUT_PTHS)

    # Llama runtime
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--master_port", type=str, default="5678")

    # Names and pth keys
    parser.add_argument("--part_id_key", type=str, default="part_category_id")
    parser.add_argument("--part_name_key", type=str, default="part_class_name")
    parser.add_argument("--expected_num_parts", type=int, default=116)
    parser.add_argument("--output_part_key", type=str, default="llama_part_ann_feats")

    # Prompt/system
    parser.add_argument("--system_template", type=str, default=DEFAULT_SYSTEM_TEMPLATE)
    parser.add_argument("--ignore_system_in_prompt_file", action="store_true")
    parser.add_argument("--bracket_cls", action="store_true", default=True, help="Use [name] when replacing [cls], matching the old script style.")
    parser.add_argument("--no_bracket_cls", dest="bracket_cls", action="store_false")

    # Generation and pooling
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_gen_len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--response_pool", choices=["mean", "last"], default="mean")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--save_half", action="store_true", default=True)
    parser.add_argument("--save_float", dest="save_half", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit_classes", type=int, default=0, help="Debug only: extract first N classes.")

    # Injection controls
    parser.add_argument("--no_inject", action="store_true", help="Only save the Llama3 bank; do not write train/val pth.")
    parser.add_argument("--replace_part_ann_feats", action="store_true", help="Also replace ann['part_ann_feats']; default keeps CLIP part feats.")
    parser.add_argument("--pth_dtype", choices=["float16", "float32"], default="float16")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if os.path.dirname(args.bank_out):
        os.makedirs(os.path.dirname(args.bank_out), exist_ok=True)

    # Infer id-ordered VOC116 part names from the same pth that will receive features.
    names = infer_part_names_from_pths(
        pth_paths=args.input_pth,
        part_id_key=args.part_id_key,
        part_name_key=args.part_name_key,
        expected_num_parts=args.expected_num_parts,
    )
    print(f"[OK] inferred {len(names)} part names from pth")
    print(f"     first 5: {names[:5]}")

    bank = extract_llama_part_bank(args, names)
    torch.save(bank, args.bank_out)
    print(f"[OK] saved Llama3 part bank to {args.bank_out}")
    print(f"     feats:      {tuple(bank['feats'].shape)}")
    print(f"     mean_feats: {tuple(bank['mean_feats'].shape)}")
    print(f"     prompts:    {len(bank['prompts'])}")

    if not args.no_inject:
        for input_pth, output_pth in parse_io_pairs(args.input_pth, args.output_pth):
            inject_part_features_into_pth(
                input_pth=input_pth,
                output_pth=output_pth,
                llama_part_feats=bank["mean_feats"],
                part_id_key=args.part_id_key,
                output_part_key=args.output_part_key,
                replace_part_ann_feats=args.replace_part_ann_feats,
                dtype=args.pth_dtype,
            )


if __name__ == "__main__":
    main()
