#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama3 text feature extraction for Talk2DINO VOC object / part text features.

Clean test16 convention:
  target=obj  -> inject ann['ann_feats']
  target=part -> inject ann['part_ann_feats']

The saved bank always contains all generated prompt/epoch features:
  bank['feats']      : [num_classes, num_prompts * num_epochs, 4096]
  bank['mean_feats'] : [num_classes, 4096]

Injection can write either all prompt/epoch features or their mean:
  --inject_mode all
    obj:  ann_feats      [K, 4096]
    part: part_ann_feats [num_parts_in_object, K, 4096]
  --inject_mode mean
    obj:  ann_feats      [4096]
    part: part_ann_feats [num_parts_in_object, 4096]
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


DEFAULT_CKPT_DIR = "Meta-Llama-3-8B-Instruct"
DEFAULT_TOKENIZER_PATH = "Meta-Llama-3-8B-Instruct/tokenizer.model"

OBJ_SYSTEM_TEMPLATE = (
    "You are a visual object description assistant. "
    "Answer only with concise visual appearance information useful for image segmentation. "
    "Focus on foreground shape, boundary, surface appearance, and visible components. "
    "Avoid function, usage, behavior, scene context, and non-visual commonsense."
)

PART_SYSTEM_TEMPLATE = (
    "You are a visual part description assistant. "
    "Answer only with concise visual appearance information useful for image segmentation. "
    "Focus on local shape, boundary, surface appearance, and visual cues. "
    "Avoid function, usage, behavior, scene context, and non-visual commonsense."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_env(master_port: str) -> None:
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


def load_prompt_file(path: str) -> Tuple[List[str], str]:
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
    return [
        {"role": "system", "content": system_template},
        {"role": "user", "content": fill_prompt(prompt_template, name, bracket_cls=bracket_cls)},
    ]


def infer_names_from_pths(
    pth_paths: Sequence[str],
    id_key: str,
    name_key: str,
    expected_num_classes: int,
) -> List[str]:
    id_to_name: Dict[int, str] = {}
    for pth_path in pth_paths:
        data = torch.load(pth_path, map_location="cpu")
        if "annotations" not in data:
            raise KeyError(f"{pth_path} has no top-level key 'annotations'.")
        for ann_idx, ann in enumerate(data["annotations"]):
            if id_key not in ann:
                continue
            if name_key not in ann:
                raise KeyError(
                    f"Annotation {ann_idx} in {pth_path} has '{id_key}' but no '{name_key}'. "
                    f"Available keys: {list(ann.keys())}"
                )
            ids = to_int_list(ann[id_key])
            names = to_str_list(ann[name_key])
            if len(ids) != len(names):
                raise ValueError(
                    f"Annotation {ann_idx} in {pth_path}: len({id_key})={len(ids)} "
                    f"but len({name_key})={len(names)}."
                )
            for cid, cname in zip(ids, names):
                if cid in id_to_name and id_to_name[cid] != cname:
                    raise ValueError(
                        f"Conflicting name for id {cid}: '{id_to_name[cid]}' vs '{cname}' "
                        f"in {pth_path}, annotation {ann_idx}."
                    )
                id_to_name[cid] = cname

    if not id_to_name:
        raise RuntimeError(f"Could not infer names from key pair ({id_key}, {name_key}).")

    max_id = max(id_to_name.keys())
    names_ordered: List[str] = []
    missing: List[int] = []
    for cid in range(max_id + 1):
        if cid not in id_to_name:
            names_ordered.append(f"__missing_{cid}__")
            missing.append(cid)
        else:
            names_ordered.append(id_to_name[cid])

    if missing:
        raise RuntimeError(f"Missing names for ids: {missing[:30]}{'...' if len(missing) > 30 else ''}.")
    if expected_num_classes > 0 and len(names_ordered) != expected_num_classes:
        raise RuntimeError(
            f"Expected {expected_num_classes} classes, but inferred {len(names_ordered)} from pth. max_id={max_id}."
        )
    return names_ordered


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
    feats = torch.stack([x.detach().float().cpu() for x in corr_feats], dim=0)
    if pool == "mean":
        return feats.mean(dim=0)
    if pool == "last":
        return feats[-1]
    raise ValueError(f"Unknown response_pool: {pool}")


def extract_llama_bank(args: argparse.Namespace, names: List[str], system_template: str) -> Dict[str, Any]:
    prompts, system_from_file = load_prompt_file(args.prompt_file)
    if system_from_file and not args.ignore_system_in_prompt_file:
        system_template = system_from_file

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
    generated_texts: List[List[List[str]]] = []

    for epoch in range(args.num_epochs):
        epoch_prompt_feats: List[torch.Tensor] = []
        epoch_generated_texts: List[List[str]] = [[] for _ in names]

        for prompt_idx, prompt_template in enumerate(prompts):
            dialogs = [build_dialog(system_template, prompt_template, name, args.bracket_cls) for name in names]
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
                    feat = pool_corr_feats(gen["corr_feats"], pool=args.response_pool)
                    if args.normalize:
                        feat = F.normalize(feat.float(), dim=0)
                    prompt_feats.append(feat.cpu())
                    epoch_generated_texts[batch_indices[local_i]].append(str(gen.get("content", "")))

            epoch_prompt_feats.append(torch.stack(prompt_feats, dim=0))

        epoch_feat_tensor = torch.stack(epoch_prompt_feats, dim=0).permute(1, 0, 2).contiguous()
        all_feats_by_epoch_prompt.append(epoch_feat_tensor)
        generated_texts.append(epoch_generated_texts)

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
            "target": args.target,
            "logic": "generated_response_corr_feats_pooling",
            "response_pool": args.response_pool,
            "num_epochs": args.num_epochs,
            "num_prompts": len(prompts),
            "num_features_per_class": args.num_epochs * len(prompts),
            "dim": dim,
            "normalize": args.normalize,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_gen_len": args.max_gen_len,
            "max_seq_len": args.max_seq_len,
            "bracket_cls": args.bracket_cls,
            "id_key": args.id_key,
            "name_key": args.name_key,
            "output_key": args.output_key,
            "inject_mode": args.inject_mode,
        },
    }


def inject_features_into_pth(
    input_pth: str,
    output_pth: str,
    bank: Dict[str, Any],
    target: str,
    id_key: str,
    output_key: str,
    inject_mode: str,
    dtype: str,
) -> None:
    data = torch.load(input_pth, map_location="cpu")
    if "annotations" not in data:
        raise KeyError(f"{input_pth} has no top-level key 'annotations'.")

    source = bank["feats"] if inject_mode == "all" else bank["mean_feats"]
    source = source.detach().cpu().float()
    out_dtype = torch.float16 if dtype == "float16" else torch.float32

    for ann_idx, ann in enumerate(tqdm(data["annotations"], desc=f"inject {Path(input_pth).name}")):
        if id_key not in ann:
            raise KeyError(f"Annotation {ann_idx} does not contain key '{id_key}'. Available keys: {list(ann.keys())}")
        ids = to_int_list(ann[id_key])
        idx = torch.as_tensor(ids, dtype=torch.long)
        if idx.numel() == 0:
            if inject_mode == "all":
                out_tensor = torch.zeros((0, source.shape[1], source.shape[2]), dtype=out_dtype)
            else:
                out_tensor = torch.zeros((0, source.shape[1]), dtype=out_dtype)
        else:
            if idx.min().item() < 0 or idx.max().item() >= source.shape[0]:
                raise IndexError(
                    f"Annotation {ann_idx} has id outside feature bank range: "
                    f"min={idx.min().item()}, max={idx.max().item()}, bank_size={source.shape[0]}"
                )
            idx = idx.to(device=source.device, dtype=torch.long)
            out_tensor = source.index_select(0, idx).to(dtype=out_dtype).cpu()

        # Object annotation has one category id; store [K,D] or [D], not [1,K,D] or [1,D].
        if target == "obj" and out_tensor.shape[0] == 1:
            out_tensor = out_tensor[0]

        ann[output_key] = out_tensor

    out_dir = os.path.dirname(output_pth)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, output_pth)
    print(f"[OK] wrote {output_pth}")
    print(f"     injected field: {output_key}")
    print(f"     inject_mode: {inject_mode}, dtype={out_dtype}")


def parse_io_pairs(input_pths: List[str], output_pths: List[str]) -> List[Tuple[str, str]]:
    if len(input_pths) != len(output_pths):
        raise ValueError("--input_pth and --output_pth must have the same number of values.")
    return list(zip(input_pths, output_pths))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Llama3 text features for VOC objects or VOC116 parts.")
    parser.add_argument("--target", choices=["obj", "part"], required=True)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--bank_out", type=str, required=True)
    parser.add_argument("--input_pth", type=str, nargs="+", required=True)
    parser.add_argument("--output_pth", type=str, nargs="+", required=True)

    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--master_port", type=str, default="5678")

    parser.add_argument("--id_key", type=str, default=None)
    parser.add_argument("--name_key", type=str, default=None)
    parser.add_argument("--expected_num_classes", type=int, default=None)
    parser.add_argument("--output_key", type=str, default=None)

    parser.add_argument("--system_template", type=str, default=None)
    parser.add_argument("--ignore_system_in_prompt_file", action="store_true")
    parser.add_argument("--bracket_cls", action="store_true", default=True)
    parser.add_argument("--no_bracket_cls", dest="bracket_cls", action="store_false")

    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--max_gen_len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--response_pool", choices=["mean", "last"], default="mean")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--save_half", action="store_true", default=True)
    parser.add_argument("--save_float", dest="save_half", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit_classes", type=int, default=0)

    parser.add_argument("--inject_mode", choices=["all", "mean"], default="all")
    parser.add_argument("--pth_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--no_inject", action="store_true")
    return parser


def apply_target_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.target == "obj":
        args.id_key = args.id_key or "category_id"
        args.name_key = args.name_key or "class_name"
        args.expected_num_classes = 20 if args.expected_num_classes is None else args.expected_num_classes
        args.output_key = args.output_key or "ann_feats"
        args.system_template = args.system_template or OBJ_SYSTEM_TEMPLATE
    else:
        args.id_key = args.id_key or "part_category_id"
        args.name_key = args.name_key or "part_class_name"
        args.expected_num_classes = 116 if args.expected_num_classes is None else args.expected_num_classes
        args.output_key = args.output_key or "part_ann_feats"
        args.system_template = args.system_template or PART_SYSTEM_TEMPLATE
    return args


def main() -> None:
    args = apply_target_defaults(build_parser().parse_args())

    out_dir = os.path.dirname(args.bank_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    names = infer_names_from_pths(
        pth_paths=args.input_pth,
        id_key=args.id_key,
        name_key=args.name_key,
        expected_num_classes=args.expected_num_classes,
    )
    print(f"[OK] inferred {len(names)} {args.target} names")
    print(f"     first 5: {names[:5]}")

    bank = extract_llama_bank(args, names, args.system_template)
    torch.save(bank, args.bank_out)
    print(f"[OK] saved bank to {args.bank_out}")
    print(f"     feats:      {tuple(bank['feats'].shape)}")
    print(f"     mean_feats: {tuple(bank['mean_feats'].shape)}")

    if not args.no_inject:
        for input_pth, output_pth in parse_io_pairs(args.input_pth, args.output_pth):
            inject_features_into_pth(
                input_pth=input_pth,
                output_pth=output_pth,
                bank=bank,
                target=args.target,
                id_key=args.id_key,
                output_key=args.output_key,
                inject_mode=args.inject_mode,
                dtype=args.pth_dtype,
            )


if __name__ == "__main__":
    main()
