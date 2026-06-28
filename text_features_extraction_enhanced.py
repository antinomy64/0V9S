import argparse
import clip
import json
import math

import os
import torch
import torchvision.transforms as T

from src.hooks import get_self_attention, process_self_attention, get_second_last_out, feats, get_clip_second_last_dense_out
from PIL import Image
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer
from src.webdatasets_util import cc2coco_format, create_webdataset_tar
from src.hooks import get_all_out_tokens, feats


def encode_caption_ensemble(model, captions, device):
    inputs = clip.tokenize(captions, truncate=True).to(device)
    with torch.no_grad():
        outputs = model.encode_text(inputs)
    feat = outputs / outputs.norm(dim=-1, keepdim=True)
    feat = feat.mean(dim=0)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.to(dtype=torch.float16, device='cpu')


def encode_part_caption_ensemble(model, part_captions, device):
    """
    part_captions: List[List[str]]
    returns: Tensor [K, C] on cpu, or empty tensor if no parts
    """
    if part_captions is None or len(part_captions) == 0:
        return torch.empty(0)

    part_feats = []
    for captions in part_captions:
        if isinstance(captions, (list, tuple)):
            feat = encode_caption_ensemble(model, list(captions), device)
        else:
            feat = encode_caption_ensemble(model, [captions], device)
        part_feats.append(feat)

    return torch.stack(part_feats, dim=0)

def infer_base_part_name(part_name, class_name=None):
    """
    Convert an object-specific part name to a part-only name.

    Examples:
      "cat's tail" -> "tail"
      "person's lower arm" -> "lower arm"
      "aeroplane wing" -> "wing"  (fallback if class_name is known)
    """
    part_name = str(part_name).strip()
    if class_name is not None:
        class_name = str(class_name).strip()
        prefixes = [
            f"{class_name}'s ",
            f"{class_name}’s ",
            f"{class_name} ",
        ]
        for prefix in prefixes:
            if part_name.startswith(prefix):
                return part_name[len(prefix):].strip()

    for sep in ["'s ", "’s "]:
        if sep in part_name:
            return part_name.split(sep, 1)[1].strip()

    return part_name


def build_part_only_prompts(obj_part_prompts, obj_part_name, base_part_name):
    """
    Build part-only prompts by replacing the object-specific part phrase
    in the original prompt ensemble.

    Example:
      "a close-up photo of cat's tail" -> "a close-up photo of tail"
    """
    if isinstance(obj_part_prompts, (list, tuple)):
        prompts = list(obj_part_prompts)
    else:
        prompts = [obj_part_prompts]

    part_only_prompts = []
    changed = False
    for prompt in prompts:
        prompt = str(prompt)
        new_prompt = prompt.replace(str(obj_part_name), str(base_part_name))
        if new_prompt != prompt:
            changed = True
        part_only_prompts.append(new_prompt)

    # Fallback: if the original prompts do not contain the exact obj-part phrase,
    # use a small stable prompt ensemble for the part-only name.
    if not changed:
        part_only_prompts = [
            f"a photo of {base_part_name}",
            f"a close-up photo of {base_part_name}",
            f"a cropped photo of {base_part_name}",
            f"a visible {base_part_name}",
            f"a photo of the {base_part_name}",
            f"a close-up photo of the {base_part_name}",
        ]

    return part_only_prompts


def build_prompt_caches(annotations):
    obj_prompt_cache = {}
    part_prompt_cache = {}
    part_only_prompt_cache = {}
    part_to_base_cache = {}

    for ann in annotations:
        class_name = ann.get('class_name', None)
        caption = ann.get('caption', None)
        if class_name is not None and caption is not None and class_name not in obj_prompt_cache:
            if isinstance(caption, (list, tuple)):
                obj_prompt_cache[class_name] = list(caption)
            else:
                obj_prompt_cache[class_name] = [caption]

        part_names = ann.get('part_class_name', []) or []
        part_captions = ann.get('part_caption', []) or []
        for part_name, prompts in zip(part_names, part_captions):
            if isinstance(prompts, (list, tuple)):
                prompts = list(prompts)
            else:
                prompts = [prompts]

            if part_name not in part_prompt_cache:
                part_prompt_cache[part_name] = prompts

            base_part_name = infer_base_part_name(part_name, class_name)
            part_to_base_cache[part_name] = base_part_name

            if base_part_name not in part_only_prompt_cache:
                part_only_prompt_cache[base_part_name] = build_part_only_prompts(
                    prompts, part_name, base_part_name
                )

    return obj_prompt_cache, part_prompt_cache, part_only_prompt_cache, part_to_base_cache

def encode_prompt_cache(model, prompt_cache, device, desc='Encoding prompt cache'):
    feat_cache = {}
    for name in tqdm(prompt_cache.keys(), desc=desc):
        feat_cache[name] = encode_caption_ensemble(model, prompt_cache[name], device)
    return feat_cache


def run_bert_extraction(model_name, ann_path, batch_size, out_path, extract_dense_out=False, extract_second_last_dense_out=False,
                          write_as_wds=False, num_shards=25, n_in_splits=4, in_batch_offset=0, out_offset=0,
                          use_caption_ensemble=False, part_residual=False, _lambda=1.0,
                          part_enhanced=False, enhanced_beta=0.5, enhanced_mode='avg', enhanced_gamma=1.0, save_components=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'bert' in model_name:
        model_type = 'bert'
        field_name = 'bert-base_features'
        model = BertModel.from_pretrained(model_name, output_hidden_states = False)
        # load the corresponding wordtokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model_type = 'clip'
        field_name = 'ann_feats'
        model, _ = clip.load(model_name)
        if extract_dense_out:
            # in this case we register a forward hook with the aim of getting all the tokens and not only the cls
            model.ln_final.register_forward_hook(get_all_out_tokens)
        if extract_second_last_dense_out:
            model.transformer.resblocks[-2].register_forward_hook(get_clip_second_last_dense_out)

    model.eval()
    model.to(device)

    if os.path.isdir(ann_path):
        # if we have a dir as path we assume that the path refere to gcc3m webdataset
        data = cc2coco_format(ann_path, n_in_splits, in_batch_offset)
    else:
        # otherwise we treat the dataset as a COCO dataset
        data = torch.load(ann_path)

    if part_enhanced and not use_caption_ensemble:
        raise ValueError("--part_enhanced currently requires --use_caption_ensemble")

    print("Starting the features extraction...")
    if part_enhanced:
        if enhanced_mode == 'avg':
            print(f"[Enhanced:avg] part_ann_feats = normalize((1 - beta) * obj_part + beta * part_only), beta={enhanced_beta}")
        elif enhanced_mode == 'orthogonal':
            print(f"[Enhanced:orthogonal] part_ann_feats = normalize(obj_part + gamma * normalize(part_only - <part_only,obj_part> obj_part)), gamma={enhanced_gamma}")
        else:
            raise ValueError(f"Unknown enhanced_mode: {enhanced_mode}")
        if save_components:
            print("[Enhanced] save_components=True: also saving part_ann_feats_objpart, part_ann_feats_partonly and part_base_class_name")
    n_capts = len(data['annotations'])
    n_batch = math.ceil(n_capts / batch_size)
    for i in tqdm(range(n_batch)):
        start = i * batch_size
        end = start + batch_size if i < n_batch - 1 else n_capts

        texts = [data['annotations'][j]['caption'] for j in range(start, end)]

        if model_type == 'bert':
            inputs = tokenizer(texts, return_tensors='pt', padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            for j in range(start, end):
                data['annotations'][j][field_name] = outputs['pooler_output'][j - start].to('cpu')

        if model_type == 'clip':
            if use_caption_ensemble:
                obj_prompt_cache, part_prompt_cache, part_only_prompt_cache, part_to_base_cache = build_prompt_caches(data['annotations'])

                obj_feat_cache = encode_prompt_cache(
                    model, obj_prompt_cache, device, desc='Encoding unique object prompts'
                )
                part_feat_cache = encode_prompt_cache(
                    model, part_prompt_cache, device, desc='Encoding unique object-specific part prompts'
                )
                part_only_feat_cache = {}
                if part_enhanced:
                    part_only_feat_cache = encode_prompt_cache(
                        model, part_only_prompt_cache, device, desc='Encoding unique part-only prompts'
                    )

                for ann in tqdm(data['annotations'], desc='Assigning cached text features'):
                        class_name = ann['class_name']
                        obj_feat = obj_feat_cache[class_name]
                        ann[field_name] = obj_feat

                        part_names = ann.get('part_class_name', []) or []
                        if len(part_names) > 0:
                            obj_part_feats = torch.stack([part_feat_cache[name] for name in part_names], dim=0)
                            part_feats = obj_part_feats

                            if part_enhanced:
                                part_only_feats = torch.stack(
                                    [part_only_feat_cache[part_to_base_cache[name]] for name in part_names],
                                    dim=0,
                                )

                                obj_part_f = obj_part_feats.float()
                                part_only_f = part_only_feats.float()
                                obj_part_f = obj_part_f / (obj_part_f.norm(dim=-1, keepdim=True) + 1e-6)
                                part_only_f = part_only_f / (part_only_f.norm(dim=-1, keepdim=True) + 1e-6)

                                if enhanced_mode == 'avg':
                                    # Simple enhanced:
                                    #   enhanced = normalize((1 - beta) * obj_part + beta * part_only)
                                    part_feats = (1.0 - enhanced_beta) * obj_part_f + enhanced_beta * part_only_f
                                elif enhanced_mode == 'orthogonal':
                                    # Orthogonal enhanced:
                                    #   delta = part_only - <part_only, obj_part> * obj_part
                                    #   enhanced = normalize(obj_part + gamma * normalize(delta))
                                    proj_scalar = (part_only_f * obj_part_f).sum(dim=-1, keepdim=True)
                                    delta = part_only_f - proj_scalar * obj_part_f
                                    delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-6)
                                    part_feats = obj_part_f + enhanced_gamma * delta
                                else:
                                    raise ValueError(f"Unknown enhanced_mode: {enhanced_mode}")

                                part_feats = part_feats / (part_feats.norm(dim=-1, keepdim=True) + 1e-6)
                                part_feats = part_feats.half()

                                if save_components:
                                    ann['part_ann_feats_objpart'] = obj_part_feats
                                    ann['part_ann_feats_partonly'] = part_only_feats
                                    ann['part_base_class_name'] = [part_to_base_cache[name] for name in part_names]

                            if part_residual:
                                part_feats = part_feats.float()
                                obj_feat = obj_feat.float()
                                
                                obj = obj_feat.squeeze(0)  # [D]
                                obj = obj / (obj.norm() + 1e-6)

                                proj_scalar = part_feats @ obj  # [K]
                                proj = proj_scalar.unsqueeze(-1) * obj.unsqueeze(0)  # [K, D]

                                part_feats = part_feats - _lambda * proj
                                part_feats = part_feats / (part_feats.norm(dim=-1, keepdim=True) + 1e-6)

                                part_feats = part_feats.half()
                                obj_feat = obj_feat.half()


                            ann['part_ann_feats'] = part_feats
                        else:
                            ann['part_ann_feats'] = obj_feat.new_zeros((0, obj_feat.shape[-1]))
                break
            else:
                inputs = clip.tokenize(texts, truncate=True).to(device)
                with torch.no_grad():
                    outputs = model.encode_text(inputs)
                    if extract_dense_out:
                        clip_txt_out_tokens = feats['clip_txt_out_tokens'] @ model.text_projection
                        masks = inputs > 0

                for j in range(start, end):
                    data['annotations'][j][field_name] = outputs[j - start].to(dtype=torch.float16, device='cpu')
                    if extract_dense_out:
                        data['annotations'][j]['clip_txt_out_tokens'] = clip_txt_out_tokens[j - start].to(dtype=torch.float16, device='cpu')
                        data['annotations'][j]['text_input_mask'] = masks[j - start].to('cpu')
                    if extract_second_last_dense_out:
                        data['annotations'][j]['clip_second_last_out'] = feats['clip_second_last_out'][j - start].to(dtype=torch.float16, device='cpu')
                        data['annotations'][j]['text_argmax'] = inputs.argmax(dim=-1)[j - start].to('cpu')

                    # minimal additive part feature extraction
                    part_caption = data['annotations'][j].get('part_caption', None)
                    if part_caption is not None:
                        part_feat = encode_part_caption_ensemble(model, part_caption, device)
                        obj_feat = data['annotations'][j]['ann_feats'].unsqueeze(0)

                        if part_residual:
                            part_feat = part_feat.float()
                            obj_feat = obj_feat.float()

                            obj = obj_feat.squeeze(0)  # [D]
                            obj = obj / (obj.norm() + 1e-6)

                            proj_scalar = part_feat @ obj  # [K]
                            proj = proj_scalar.unsqueeze(-1) * obj.unsqueeze(0)  # [K, D]

                            part_feat = part_feat - _lambda * proj
                            part_feat = part_feat / (part_feat.norm(dim=-1, keepdim=True) + 1e-6)
                            
                            part_feat = part_feat.half()
                            obj_feat = obj_feat.half()
                            
                        data['annotations'][j]['part_ann_feats'] = part_feat

    print("Feature extraction done!")

    if write_as_wds:
        os.makedirs(out_path, exist_ok=True)
        create_webdataset_tar(data, out_path, num_shards, out_offset)
    else:
        if out_path is None:
            # we use as output path the ann_path but with the extension pth
            out_path = os.path.splitext(ann_path)[0] + '.pth' 
        torch.save(data, out_path)
    print(f"Features saved at {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, default="coco/test1k.json", help="Directory of the annotation file") 
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--model', type=str, default="ViT-B/16", help="Model configuration to extract features from")
    parser.add_argument('--out_path', type=str, default=None, help="Pth of the output file, if setted to None. out_pat = ann_path") 
    parser.add_argument('--extract_dense_out', action="store_true", default=False, help="If setted, all the token of the last layer of CLIP will be extracted")
    parser.add_argument('--extract_second_last_dense_out', action="store_true", default=False, help="If setted, the second last output of the model will be extracted")
    parser.add_argument('--write_as_wds', action="store_true", default=False, help="If setted, the output will be written as a webdataset") 
    parser.add_argument('--n_shards', type=int, default=10, help="Number of shards in which the webdataset is splitted. Only relevant if --write_as_wds is setted.")
    parser.add_argument('--n_in_splits', type=int, default=1, help="Number of splits in which we want to divide the tar files. For example, with 4 n_split we elaborate 332 // 4 = 83 tar files.")
    parser.add_argument('--in_batch_offset', type=int, default=0, help="Of the n_splits in which we have divided tars, we decide which of them elaborate")
    parser.add_argument('--out_offset', type=int, default=0, help="Index of the first shard to save")
    parser.add_argument('--use_caption_ensemble', action='store_true', default=False, help='If set, ann["caption"] can be a list of captions and their text features are averaged')
    parser.add_argument('--part_residual', action='store_true')
    parser.add_argument('--_lambda', type=float, default=1.0)
    parser.add_argument('--part_enhanced', action='store_true', help='If set, use enhanced part text. Requires --use_caption_ensemble.')
    parser.add_argument('--enhanced_mode', type=str, default='avg', choices=['avg', 'orthogonal'], help='avg: normalize((1-beta)*obj_part + beta*part_only); orthogonal: add only the part-only direction orthogonal to obj_part.')
    parser.add_argument('--enhanced_beta', type=float, default=0.5, help='For --enhanced_mode avg: weight for part-only feature.')
    parser.add_argument('--enhanced_gamma', type=float, default=1.0, help='For --enhanced_mode orthogonal: strength for the orthogonal part-only direction.')
    parser.add_argument('--save_components', action='store_true', help='If set with --part_enhanced, save obj-part and part-only component features for debugging.')
    args = parser.parse_args()

    run_bert_extraction(args.model, args.ann_path, args.batch_size, args.out_path, args.extract_dense_out, args.extract_second_last_dense_out,
                        args.write_as_wds, args.n_shards, args.n_in_splits, args.in_batch_offset, args.out_offset,
                        args.use_caption_ensemble, args.part_residual, args._lambda,
                        args.part_enhanced, args.enhanced_beta, args.enhanced_mode, args.enhanced_gamma, args.save_components)
if __name__ == '__main__':
    main()
