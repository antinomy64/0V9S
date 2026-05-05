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


def run_bert_extraction(model_name, ann_path, batch_size, out_path, extract_dense_out=False, extract_second_last_dense_out=False,
                          write_as_wds=False, num_shards=25, n_in_splits=4, in_batch_offset=0, out_offset=0,
                          use_caption_ensemble=False):
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

    print("Starting the features extraction...")
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
                obj_prompt_cache, part_prompt_cache = build_prompt_caches(data['annotations'])

                obj_feat_cache = encode_prompt_cache(
                    model, obj_prompt_cache, device, desc='Encoding unique object prompts'
                )
                part_feat_cache = encode_prompt_cache(
                    model, part_prompt_cache, device, desc='Encoding unique part prompts'
                )

                for ann in tqdm(data['annotations'], desc='Assigning cached text features'):
                    class_name = ann['class_name']
                    ann[field_name] = obj_feat_cache[class_name]

                    part_names = ann.get('part_class_name', []) or []
                    if len(part_names) > 0:
                        ann['part_ann_feats'] = torch.stack(
                            [part_feat_cache[name] for name in part_names], dim=0
                        )
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
                        data['annotations'][j]['part_ann_feats'] = encode_part_caption_ensemble(model, part_caption, device)

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
    args = parser.parse_args()

    run_bert_extraction(args.model, args.ann_path, args.batch_size, args.out_path, args.extract_dense_out, args.extract_second_last_dense_out,
                        args.write_as_wds, args.n_shards, args.n_in_splits, args.in_batch_offset, args.out_offset,
                        args.use_caption_ensemble)
if __name__ == '__main__':
    main()
