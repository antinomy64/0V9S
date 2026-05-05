import argparse
import clip
import math
import os
import torch

from src.hooks import get_clip_second_last_dense_out, get_all_out_tokens, feats
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer
from src.webdatasets_util import cc2coco_format, create_webdataset_tar


def encode_caption_ensemble(model, captions, device):
    inputs = clip.tokenize(captions, truncate=True).to(device)
    with torch.no_grad():
        outputs = model.encode_text(inputs)
    feat = outputs / outputs.norm(dim=-1, keepdim=True)
    feat = feat.mean(dim=0)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.to(dtype=torch.float16, device='cpu')


def encode_prompt_list(model, prompts, device):
    if prompts is None or len(prompts) == 0:
        return torch.empty(0)
    inputs = clip.tokenize(list(prompts), truncate=True).to(device)
    with torch.no_grad():
        outputs = model.encode_text(inputs)
    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
    return outputs.to(dtype=torch.float16, device='cpu')


def encode_part_caption_ensemble(model, part_captions, device):
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


def build_prompt_caches(annotations):
    obj_prompt_cache = {}
    part_prompt_cache = {}

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
            if part_name not in part_prompt_cache:
                if isinstance(prompts, (list, tuple)):
                    part_prompt_cache[part_name] = list(prompts)
                else:
                    part_prompt_cache[part_name] = [prompts]

    return obj_prompt_cache, part_prompt_cache


def encode_prompt_cache(model, prompt_cache, device, desc='Encoding prompt cache'):
    feat_cache = {}
    for name in tqdm(prompt_cache.keys(), desc=desc):
        feat_cache[name] = encode_caption_ensemble(model, prompt_cache[name], device)
    return feat_cache


def parse_generic_part_name(part_name: str) -> str:
    if part_name is None:
        return ''
    if "'s " in part_name:
        return part_name.split("'s ", 1)[1].strip()
    return str(part_name).strip()


def normalize_cpu_feat(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return x.to(dtype=torch.float16, device='cpu')


def build_class_to_part_prompt_lists(annotations):
    class_to_part_prompts = {}
    for ann in annotations:
        class_name = ann.get('class_name', None)
        part_names = ann.get('part_class_name', []) or []
        part_captions = ann.get('part_caption', []) or []
        if class_name is None:
            continue

        for part_name, prompts in zip(part_names, part_captions):
            if class_name not in class_to_part_prompts:
                class_to_part_prompts[class_name] = {}
            prompt_list = list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]
            if part_name not in class_to_part_prompts[class_name]:
                class_to_part_prompts[class_name][part_name] = prompt_list
    return class_to_part_prompts


def make_generic_prompts_from_specific_prompts(specific_prompts, class_specific_part_name, generic_part_name):
    generic_prompts = []
    for prompt in specific_prompts:
        prompt = str(prompt)
        if class_specific_part_name in prompt:
            generic_prompts.append(prompt.replace(class_specific_part_name, generic_part_name))
        else:
            generic_prompts.append(f'a photo of {generic_part_name}')
    return generic_prompts



def build_composed_obj_feat_cache_from_all_parts(model, annotations, device):
    """
    Build a composed object feature from ALL parts of each object class in the dataset.

    For each object class c:
      1) collect all class-specific parts p belonging to c over the whole dataset
      2) encode the full prompt list of each part and mean over prompts:
           sbar_{c,p} = mean_i s_{c,p,i}
      3) mean over all parts of the class:
           o_c = mean_p sbar_{c,p}

    Returns:
      obj_feat_cache[class_name] = normalized composed object feature
    """
    class_to_part_prompts = build_class_to_part_prompt_lists(annotations)

    specific_part_mean_cache = {}
    for class_name, part_dict in tqdm(class_to_part_prompts.items(), desc='Encoding all-part means for composed obj features'):
        for class_specific_part_name, specific_prompts in part_dict.items():
            feats_sp = encode_prompt_list(model, specific_prompts, device).float()
            specific_part_mean_cache[class_specific_part_name] = normalize_cpu_feat(feats_sp.mean(dim=0))

    obj_feat_cache = {}
    for class_name, part_dict in tqdm(class_to_part_prompts.items(), desc='Building composed object features from all parts'):
        if len(part_dict) == 0:
            continue
        part_mean_feats = [specific_part_mean_cache[name].float() for name in part_dict.keys()]
        obj_feat = torch.stack(part_mean_feats, dim=0).mean(dim=0)
        obj_feat_cache[class_name] = normalize_cpu_feat(obj_feat)

    return obj_feat_cache

def build_mean_residual_pairwise_caches(model, annotations, device, subtract_obj_feat: bool = True,):
    """
    User-requested version:

    1) Encode all 80 class-specific obj-part prompts:
         e.g. "a photo of cat's tail"
    2) Derive and encode matching 80 generic part prompts:
         e.g. "a photo of tail"
    3) Object feature:
         sbar_{c,p} = mean_i s_{c,p,i}
         o_c = mean_p sbar_{c,p}
    4) Enhanced part feature:
         m_{c,p} = mean_i ((s_{c,p,i} + g_{p,i}) / 2)
    5) Residual enhanced part feature:
         r_{c,p} = normalize(m_{c,p} - o_c)
    """
    class_to_part_prompts = build_class_to_part_prompt_lists(annotations)

    specific_prompt_feat_cache = {}
    generic_prompt_feat_cache = {}

    for class_name, part_dict in tqdm(class_to_part_prompts.items(), desc='Encoding class-specific/generic prompt lists'):
        for class_specific_part_name, specific_prompts in part_dict.items():
            generic_part_name = parse_generic_part_name(class_specific_part_name)
            generic_prompts = make_generic_prompts_from_specific_prompts(
                specific_prompts, class_specific_part_name, generic_part_name
            )
            specific_feats = encode_prompt_list(model, specific_prompts, device).float()
            generic_feats = encode_prompt_list(model, generic_prompts, device).float()
            L = min(specific_feats.shape[0], generic_feats.shape[0])
            specific_prompt_feat_cache[class_specific_part_name] = specific_feats[:L]
            generic_prompt_feat_cache[class_specific_part_name] = generic_feats[:L]

    specific_part_mean_cache = {}
    for _, part_dict in class_to_part_prompts.items():
        for class_specific_part_name in part_dict.keys():
            feats_sp = specific_prompt_feat_cache[class_specific_part_name]
            specific_part_mean_cache[class_specific_part_name] = normalize_cpu_feat(feats_sp.mean(dim=0))

    obj_feat_cache = {}
    for class_name, part_dict in tqdm(class_to_part_prompts.items(), desc='Building composed object features'):
        part_mean_feats = [specific_part_mean_cache[name].float() for name in part_dict.keys()]
        obj_feat = torch.stack(part_mean_feats, dim=0).mean(dim=0)
        obj_feat_cache[class_name] = normalize_cpu_feat(obj_feat)

    part_feat_cache = {}
    for class_name, part_dict in tqdm(class_to_part_prompts.items(), desc='Building pairwise residual-enhanced part features'):
        obj_feat = obj_feat_cache[class_name].float()
        for class_specific_part_name in part_dict.keys():
            s_feats = specific_prompt_feat_cache[class_specific_part_name].float()
            g_feats = generic_prompt_feat_cache[class_specific_part_name].float()
            L = min(s_feats.shape[0], g_feats.shape[0])
            enhanced = ((s_feats[:L] + g_feats[:L]) / 2.0).mean(dim=0)
            if subtract_obj_feat:
                residual = enhanced - obj_feat
            else:
                residual = enhanced
            part_feat_cache[class_specific_part_name] = normalize_cpu_feat(residual)

    return obj_feat_cache, part_feat_cache


def run_bert_extraction(model_name, ann_path, batch_size, out_path, extract_dense_out=False, extract_second_last_dense_out=False,
                          write_as_wds=False, num_shards=25, n_in_splits=4, in_batch_offset=0, out_offset=0,
                          use_caption_ensemble=False, use_mean_residual_pairwise=False, use_composed_obj_from_all_parts=False,
                          subtract_obj_feat=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'bert' in model_name:
        model_type = 'bert'
        field_name = 'bert-base_features'
        model = BertModel.from_pretrained(model_name, output_hidden_states=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model_type = 'clip'
        field_name = 'ann_feats'
        model, _ = clip.load(model_name)
        if extract_dense_out:
            model.ln_final.register_forward_hook(get_all_out_tokens)
        if extract_second_last_dense_out:
            model.transformer.resblocks[-2].register_forward_hook(get_clip_second_last_dense_out)

    model.eval()
    model.to(device)

    if os.path.isdir(ann_path):
        data = cc2coco_format(ann_path, n_in_splits, in_batch_offset)
    else:
        data = torch.load(ann_path)

    print('Starting the features extraction...')
    n_capts = len(data['annotations'])
    n_batch = math.ceil(n_capts / batch_size)

    composed_obj_feat_cache = None
    fallback_obj_feat_cache = None
    if model_type == 'clip' and use_composed_obj_from_all_parts:
        composed_obj_feat_cache = build_composed_obj_feat_cache_from_all_parts(model, data['annotations'], device)
        obj_prompt_cache, _ = build_prompt_caches(data['annotations'])
        fallback_obj_feat_cache = encode_prompt_cache(
            model, obj_prompt_cache, device, desc='Encoding fallback object prompts for composed obj branch'
        )

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
            if use_mean_residual_pairwise:
                obj_feat_cache, part_feat_cache = build_mean_residual_pairwise_caches(
                    model, data['annotations'], device, subtract_obj_feat=subtract_obj_feat
                )
                obj_prompt_cache, _ = build_prompt_caches(data['annotations'])
                fallback_obj_feat_cache = encode_prompt_cache(
                    model, obj_prompt_cache, device, desc='Encoding fallback object prompts'
                )

                for ann in tqdm(data['annotations'], desc='Assigning pairwise-80 enhanced part text features'):
                    class_name = ann['class_name']

                    # Object feature policy:
                    #   - default: keep the original object prompt logic
                    #   - if --use_composed_obj_from_all_parts is set: replace obj feature
                    #     with the composed feature built from ALL parts of that class
                    if use_composed_obj_from_all_parts:
                        ann[field_name] = obj_feat_cache.get(class_name, fallback_obj_feat_cache[class_name])
                    else:
                        ann[field_name] = fallback_obj_feat_cache[class_name]

                    part_names = ann.get('part_class_name', []) or []
                    part_category_ids = ann.get('part_category_id', []) or []
                    part_captions = ann.get('part_caption', []) or []

                    valid_parts = []
                    valid_part_category_ids = []
                    valid_part_captions = []
                    valid_part_feats = []

                    for idx_p, part_name in enumerate(part_names):
                        if part_name not in part_feat_cache:
                            continue
                        valid_parts.append(part_name)
                        if idx_p < len(part_category_ids):
                            valid_part_category_ids.append(part_category_ids[idx_p])
                        if idx_p < len(part_captions):
                            valid_part_captions.append(part_captions[idx_p])
                        valid_part_feats.append(part_feat_cache[part_name])

                    ann['part_class_name'] = valid_parts
                    if 'part_category_id' in ann:
                        ann['part_category_id'] = valid_part_category_ids
                    if 'part_caption' in ann:
                        ann['part_caption'] = valid_part_captions

                    if len(valid_part_feats) > 0:
                        ann['part_ann_feats'] = torch.stack(valid_part_feats, dim=0)
                    else:
                        ann['part_ann_feats'] = ann[field_name].new_zeros((0, ann[field_name].shape[-1]))
                break

            elif use_caption_ensemble:
                obj_prompt_cache, part_prompt_cache = build_prompt_caches(data['annotations'])

                obj_feat_cache = encode_prompt_cache(model, obj_prompt_cache, device, desc='Encoding unique object prompts')
                part_feat_cache = encode_prompt_cache(model, part_prompt_cache, device, desc='Encoding unique part prompts')

                for ann in tqdm(data['annotations'], desc='Assigning cached text features'):
                    class_name = ann['class_name']
                    if composed_obj_feat_cache is not None:
                        ann[field_name] = composed_obj_feat_cache.get(class_name, fallback_obj_feat_cache[class_name])
                    else:
                        ann[field_name] = obj_feat_cache[class_name]

                    part_names = ann.get('part_class_name', []) or []
                    if len(part_names) > 0:
                        ann['part_ann_feats'] = torch.stack([part_feat_cache[name] for name in part_names], dim=0)
                    else:
                        ann['part_ann_feats'] = ann[field_name].new_zeros((0, ann[field_name].shape[-1]))
                break
            else:
                inputs = clip.tokenize(texts, truncate=True).to(device)
                with torch.no_grad():
                    outputs = model.encode_text(inputs)
                    if extract_dense_out:
                        clip_txt_out_tokens = feats['clip_txt_out_tokens'] @ model.text_projection
                        masks = inputs > 0

                for j in range(start, end):
                    class_name = data['annotations'][j]['class_name']
                    if composed_obj_feat_cache is not None:
                        data['annotations'][j][field_name] = composed_obj_feat_cache.get(
                            class_name, fallback_obj_feat_cache[class_name]
                        )
                    else:
                        data['annotations'][j][field_name] = outputs[j - start].to(dtype=torch.float16, device='cpu')
                    if extract_dense_out:
                        data['annotations'][j]['clip_txt_out_tokens'] = clip_txt_out_tokens[j - start].to(dtype=torch.float16, device='cpu')
                        data['annotations'][j]['text_input_mask'] = masks[j - start].to('cpu')
                    if extract_second_last_dense_out:
                        data['annotations'][j]['clip_second_last_out'] = feats['clip_second_last_out'][j - start].to(dtype=torch.float16, device='cpu')
                        data['annotations'][j]['text_argmax'] = inputs.argmax(dim=-1)[j - start].to('cpu')

                    part_caption = data['annotations'][j].get('part_caption', None)
                    if part_caption is not None:
                        data['annotations'][j]['part_ann_feats'] = encode_part_caption_ensemble(model, part_caption, device)

    print('Feature extraction done!')

    if write_as_wds:
        os.makedirs(out_path, exist_ok=True)
        create_webdataset_tar(data, out_path, num_shards, out_offset)
    else:
        if out_path is None:
            out_path = os.path.splitext(ann_path)[0] + '.pth'
        torch.save(data, out_path)
    print(f'Features saved at {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, default='coco/test1k.json', help='Directory of the annotation file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--model', type=str, default='ViT-B/16', help='Model configuration to extract features from')
    parser.add_argument('--out_path', type=str, default=None, help='Pth of the output file, if setted to None. out_pat = ann_path')
    parser.add_argument('--extract_dense_out', action='store_true', default=False, help='If setted, all the token of the last layer of CLIP will be extracted')
    parser.add_argument('--extract_second_last_dense_out', action='store_true', default=False, help='If setted, the second last output of the model will be extracted')
    parser.add_argument('--write_as_wds', action='store_true', default=False, help='If setted, the output will be written as a webdataset')
    parser.add_argument('--n_shards', type=int, default=10, help='Number of shards in which the webdataset is splitted.')
    parser.add_argument('--n_in_splits', type=int, default=1, help='Number of splits in which we want to divide the tar files.')
    parser.add_argument('--in_batch_offset', type=int, default=0, help='Of the n_splits in which we have divided tars, we decide which of them elaborate')
    parser.add_argument('--out_offset', type=int, default=0, help='Index of the first shard to save')
    parser.add_argument('--use_caption_ensemble', action='store_true', default=False, help='If set, ann["caption"] can be a list of captions and their text features are averaged')
    parser.add_argument('--use_mean_residual_pairwise', action='store_true', default=False, help='Build ENHANCED part features from pairwise mean((obj-part_i + part_i)/2). Object feature stays on the original logic unless --use_composed_obj_from_all_parts is also set')
    parser.add_argument('--use_composed_obj_from_all_parts', action='store_true', default=False, help='Compose object text feature from ALL parts of that class across the dataset: mean prompts per part, then mean over all parts of the class')
    parser.add_argument(
    '--subtract_obj_feat',
    action='store_true',
    default=False,
    help='If set, subtract the composed object feature when forming the pairwise enhanced part feature'
)
    args = parser.parse_args()

    run_bert_extraction(
        args.model, args.ann_path, args.batch_size, args.out_path,
        args.extract_dense_out, args.extract_second_last_dense_out,
        args.write_as_wds, args.n_shards, args.n_in_splits,
        args.in_batch_offset, args.out_offset,
        args.use_caption_ensemble, args.use_mean_residual_pairwise,
        args.use_composed_obj_from_all_parts, args.subtract_obj_feat,
    )


if __name__ == '__main__':
    main()
