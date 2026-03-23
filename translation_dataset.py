import io
from logging import warning
from typing import Union, List
from site import PREFIXES
import warnings
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import random
import re
import matplotlib.pyplot as plt
import random as rd
import copy



def gen_prompt_uniform(
    templates, names, nouns_dict, N, symmetric, prefixes=None, abc=False
):
    nb_gen = 0
    ioi_prompts = []
    while nb_gen < N:
        temp = rd.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        name_3 = ""
        while len(set([name_1, name_2, name_3])) < 3:
            name_1 = rd.choice(names)
            name_2 = rd.choice(names)
            name_3 = rd.choice(names)

        nouns = {}
        ioi_prompt = {}
        for k in nouns_dict:
            nouns[k] = rd.choice(nouns_dict[k])
            ioi_prompt[k] = nouns[k]
        prompt = temp
        for k in nouns_dict:
            prompt = prompt.replace(k, nouns[k])

        if prefixes is not None:
            L = rd.randint(30, 40)
            pref = ".".join(rd.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[A]", name_1)
        prompt1 = prompt1.replace("[B]", name_2)
        if abc:
            prompt1 = prompt1.replace("[C]", name_3)
        prompt1 = pref + prompt1
        ioi_prompt["text"] = prompt1
        ioi_prompt["IO"] = name_1
        ioi_prompt["S"] = name_2
        ioi_prompt["TEMPLATE_IDX"] = temp_id
        ioi_prompts.append(ioi_prompt)
        if abc:
            ioi_prompts[-1]["C"] = name_3

        nb_gen += 1

        if symmetric and nb_gen < N:
            prompt2 = prompt.replace("[A]", name_2)
            prompt2 = prompt2.replace("[B]", name_1)
            prompt2 = pref + prompt2
            ioi_prompts.append(
                {"text": prompt2, "IO": name_2, "S": name_1, "TEMPLATE_IDX": temp_id}
            )
            nb_gen += 1
    return ioi_prompts


def gen_flipped_prompts(prompts, names, flip=("S2", "IO")):
    """_summary_

    Args:
        prompts (List[D]): _description_
        flip (tuple, optional): First element is the string to be replaced, Second is what to replace with. Defaults to ("S2", "IO").

    Returns:
        _type_: _description_
    """
    flipped_prompts = []

    for prompt in prompts:
        t = prompt["text"].split(" ")
        prompt = prompt.copy()
        if flip[0] == "S2":
            if flip[1] == "IO":
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = prompt["IO"]
                temp = prompt["IO"]
                prompt["IO"] = prompt["S"]
                prompt["S"] = temp
            elif flip[1] == "RAND":
                rand_name = names[np.random.randint(len(names))]
                while rand_name == prompt["IO"] or rand_name == prompt["S"]:
                    rand_name = names[np.random.randint(len(names))]
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = rand_name
            else:
                raise ValueError("Invalid flip[1] value")

        elif flip[0] == "IO":
            if flip[1] == "RAND":
                rand_name = names[np.random.randint(len(names))]
                while rand_name == prompt["IO"] or rand_name == prompt["S"]:
                    rand_name = names[np.random.randint(len(names))]

                t[t.index(prompt["IO"])] = rand_name
                t[t.index(prompt["IO"])] = rand_name
                prompt["IO"] = rand_name
            elif flip[1] == "ANIMAL":
                rand_animal = ANIMALS[np.random.randint(len(ANIMALS))]
                t[t.index(prompt["IO"])] = rand_animal
                prompt["IO"] = rand_animal
                # print(t)
            elif flip[1] == "S1":
                io_index = t.index(prompt["IO"])
                s1_index = t.index(prompt["S"])
                io = t[io_index]
                s1 = t[s1_index]
                t[io_index] = s1
                t[s1_index] = io
            else:
                raise ValueError("Invalid flip[1] value")

        elif flip[0] in ["S", "S1"]:
            if flip[1] == "ANIMAL":
                new_s = ANIMALS[np.random.randint(len(ANIMALS))]
            if flip[1] == "RAND":
                new_s = names[np.random.randint(len(names))]
            t[t.index(prompt["S"])] = new_s
            if flip[0] == "S":  # literally just change the first S if this is S1
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = new_s
                prompt["S"] = new_s
        elif flip[0] == "END":
            if flip[1] == "S":
                t[len(t) - t[::-1].index(prompt["IO"]) - 1] = prompt["S"]
        elif flip[0] == "PUNC":
            n = []

            # separate the punctuation from the words
            for i, word in enumerate(t):
                if "." in word:
                    n.append(word[:-1])
                    n.append(".")
                elif "," in word:
                    n.append(word[:-1])
                    n.append(",")
                else:
                    n.append(word)

            # remove punctuation, important that you check for period first
            if flip[1] == "NONE":
                if "." in n:
                    n[n.index(".")] = ""
                elif "," in n:
                    n[len(n) - n[::-1].index(",") - 1] = ""

            # remove empty strings
            while "" in n:
                n.remove("")

            # add punctuation back to the word before it
            while "," in n:
                n[n.index(",") - 1] += ","
                n.remove(",")

            while "." in n:
                n[n.index(".") - 1] += "."
                n.remove(".")

            t = n

        elif flip[0] == "C2":
            if flip[1] == "A":
                t[len(t) - t[::-1].index(prompt["C"]) - 1] = prompt["A"]
        elif flip[0] == "S+1":
            if t[t.index(prompt["S"]) + 1] == "and":
                t[t.index(prompt["S"]) + 1] = [
                    "with one friend named",
                    "accompanied by",
                ][np.random.randint(2)]
            else:
                t[t.index(prompt["S"]) + 1] = (
                    t[t.index(prompt["S"])]
                    + ", after a great day, "
                    + t[t.index(prompt["S"]) + 1]
                )
                del t[t.index(prompt["S"])]
        else:
            raise ValueError(f"Invalid flipper {flip[0]}")

        if "IO" in prompt:
            prompt["text"] = " ".join(t)
            flipped_prompts.append(prompt)
        else:
            flipped_prompts.append(
                {
                    "A": prompt["A"],
                    "B": prompt["B"],
                    "C": prompt["C"],
                    "text": " ".join(t),
                }
            )

    return flipped_prompts


# *Tok Idxs Methods

def get_end_idxs(toks, attn_mask):
    """
    获取每个序列中最后一个有效 token（非 pad_token_id）的索引。

    Args:
        tokenizer: 包含 pad_token_id 的 tokenizer 对象。
        toks: 一个 2D Tensor，包含多个序列的 token id。

    Returns:
        List[int]: 每个序列中最后一个有效 token 的索引。
    """
    # To get the last valid token index for each example
    end_idxs = []

    for i in range(toks.shape[0]):
        # Find the index of the last 1 in the attention_mask for each sample
        end_idxs.append((attn_mask[i] == 1).nonzero(as_tuple=True)[0].max().item() - 1)


    # print(f'[DEBUG] end_idxs[0]: {end_idxs[0]}')
    # print(f'[DEBUG] toks[0]: {toks[0]}')
    # print(f'[DEBUG] end_idxs length: {len(end_idxs)}')
    return torch.tensor(end_idxs)

def get_tgt_idxs(toks, attn_mask):
    """
    获取每个序列中最后一个有效 token（非 pad_token_id）的索引。

    Args:
        tokenizer: 包含 pad_token_id 的 tokenizer 对象。
        toks: 一个 2D Tensor，包含多个序列的 token id。

    Returns:
        List[int]: 每个序列中最后一个有效 token 的索引。
    """
    # To get the last valid token index for each example
    tgt_idxs = []

    for i in range(toks.shape[0]):
        # Find the index of the last 1 in the attention_mask for each sample
        tgt_idxs.append((attn_mask[i] == 1).nonzero(as_tuple=True)[0].max().item())


    # print(f'[DEBUG] tgt_idxs[0]: {tgt_idxs[0]}')
    # print(f'[DEBUG] toks[0]: {toks[0]}')
    # print(f'[DEBUG] tgt_idxs length: {len(tgt_idxs)}')
    return torch.tensor(tgt_idxs)

def get_src_token_idxs(toks, tokenizer):
    result = []
    # 遍历每一行
    for row in toks:
        # print(f'[DEBUG] row: {row}')
        try:
            first_quote = tokenizer.encode('"a', add_special_tokens=False)[0]
            second_quote = tokenizer.encode('a"', add_special_tokens=False)[-1]
            # 找到第一个22和第一个113的索引
            idx_22 = (row == first_quote).nonzero(as_tuple=True)[0][0].item()
            idx_113 = (row == second_quote).nonzero(as_tuple=True)[0][0].item()
            
            # 获取22和113之间的索引
            indices_between = list(range(idx_22 + 1, idx_113))
            result.append(indices_between)
        except IndexError:
            # 如果没有找到22或113则跳过该行
            raise IndexError
    return result

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_idx_dict(toks=None, attn_mask=None, tokenizer=None):

    end_idxs = get_end_idxs(
        toks=toks,
        attn_mask=attn_mask
    )
    
    src_idxs = get_src_token_idxs(toks=toks, tokenizer=tokenizer)
    
    # translation_word_ids = [item['out_token_id'] for item in preprocess_df_trans]
    
    tgt_idxs = get_tgt_idxs(
        toks=toks,
        attn_mask=attn_mask
    )
    
    return {
        "end": end_idxs,
        "src": src_idxs,
        "tgt": tgt_idxs,
        "starts": torch.zeros_like(end_idxs),
        # "translation_ids": torch.tensor(translation_word_ids)
    }


def _construct_new_translation_prompts(src_lang, tgt_lang):
    prefix = "./data/langs"
    df_src = pd.read_csv(f'{prefix}/{src_lang}/clean_llama2_{src_lang}-{tgt_lang}.csv').reindex()

    df_trans = df_src.copy()
    df_trans.rename(
        columns={
            'word_original': src_lang,
            'word_translation': tgt_lang,
        }, 
        inplace=True
    )
    # print(f'final length of df_trans: {len(df_trans)}')
    
    return df_trans

def _construct_translation_prompts(src_lang, tgt_lang, tokenizer, single_token_only=False, multi_token_only=False):
    prefix = "./data/langs/"
    df_src = pd.read_csv(f'{prefix}{src_lang}/clean.csv').reindex()
    df_tgt = pd.read_csv(f'{prefix}{tgt_lang}/clean.csv').reindex()


    count = 0
    for idx, word in enumerate(df_tgt['word_translation']):
        if word in tokenizer.get_vocab() or '▁'+word in tokenizer.get_vocab():
            count += 1
            if multi_token_only:
                df_tgt.drop(idx, inplace=True)
        elif single_token_only:
            df_tgt.drop(idx, inplace=True)

    # print(f'for {tgt_lang} {count} of {len(df_tgt)} are single tokens')

    if src_lang == tgt_lang:
        df_trans = df_tgt.copy()
        df_trans.rename(columns={'word_original': 'en', 
                                    f'word_translation': tgt_lang if tgt_lang != 'en' else 'en_tgt'}, 
                                    inplace=True)
    else:
        df_trans = df_tgt.merge(df_src, on=['word_original'], suffixes=(f'_{tgt_lang}', f'_{src_lang}'))
        
        df_trans.rename(columns={'word_original': 'en', 
                                    f'word_translation_{tgt_lang}': tgt_lang if tgt_lang != 'en' else 'en_tgt', 
                                    f'word_translation_{src_lang}': src_lang if src_lang != 'en' else 'en_in'}, 
                                    inplace=True)
    # delete all rows where en is contained in de or fr
    # if tgt_lang != 'en':
    #     for i, row in df_trans.iterrows():
    #         if row['en'].lower() in row[tgt_lang].lower():
    #             df_trans.drop(i, inplace=True)
    # print(f'df_src length: {len(df_src)}')
    # print(f'df_tgt length: {len(df_tgt)}')
    # print(f'final length of df_trans: {len(df_trans)}')
    
    return df_trans

def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n+1)]
    return tokens 

def add_spaces(tokens):
    return ['▁' + t for t in tokens] + tokens

def capitalizations(tokens):
    capitalized_tokens = []
    for token in tokens:
        # Add the original token
        capitalized_tokens.append(token)
        # Capitalize the first letter of the token if it is alphabetic
        if token and token[0].isalpha():
            capitalized_tokens.append(token[0].upper() + token[1:])
    return list(set(capitalized_tokens))

def unicode_prefix_tokid(zh_char = "云", tokenizer=None):
    start = zh_char.encode().__str__()[2:-1].split('\\x')[1]
    unicode_format = '<0x%s>'
    start_key = unicode_format%start.upper()
    if start_key in tokenizer.get_vocab():
        return tokenizer.get_vocab()[start_key]
    return None

# def process_tokens(token_str: str, tokenizer, lang):
#     with_prefixes = token_prefixes(token_str)
#     with_spaces = add_spaces(with_prefixes)
#     with_capitalizations = list(set(with_spaces))
#     # with_capitalizations = capitalizations(with_spaces)
#     # print(f'[DEBUG] with_capitalizations: {with_capitalizations}')
#     final_tokens = []
#     for tok in with_capitalizations:
#         if tok in tokenizer.get_vocab():
#             final_tokens.append(tokenizer.get_vocab()[tok])
#     if lang in ['zh', 'ru']:
#         tokid = unicode_prefix_tokid(token_str, tokenizer)
#         if tokid is not None:
#             final_tokens.append(tokid)
#     return final_tokens

# only the single token and its capitalized version
def process_tokens(token_str: str, tokenizer, lang):
    final_tokens = set()
    if lang == 'en':
        with_capitalizations = capitalizations([token_str])
    else:
        with_capitalizations = [token_str]
    for tok in with_capitalizations:
        if tok in tokenizer.get_vocab():
            final_tokens.add(tokenizer.get_vocab()[tok])
        elif '▁' + tok in tokenizer.get_vocab():
            final_tokens.add(tokenizer.get_vocab()['▁' + tok])
    return list(final_tokens)

def get_tokens(token_ids, id2voc):
    return [id2voc[tokid] for tokid in token_ids]

def compute_entropy(probas):
    return (-probas*torch.log2(probas)).sum(dim=-1)

LANG2NAME = {'fr': 'Français', 'de': 'Deutsch', 'ru': 'Русский', 'en': 'English', 'zh': '中文'}

TRANSLATION_PROMPTS = [
    '{src_lang}: "{src_word}" - {tgt_lang}: "{tgt_word}',
    # 'Translate "{src_word}" into {tgt_lang}: "',
    # 'Translate the {src_lang} word "{src_word}" to {tgt_lang}: "',
    # 'From {src_lang}: "{src_word}" to {tgt_lang}: "',
    # 'Provide the translation of "{src_word}" from {src_lang} to {tgt_lang}: "',
    # 'Q: What is "{src_word}" in {tgt_lang}? A: "',
    # 'Q: What is the {tgt_lang} translation "{src_word}" ? A: "',
    # 'Q: How do you say "{src_word}" in {tgt_lang}? A: "',
    # 'Q: What is "{src_word}" translated into {tgt_lang}? A: "'
]

FLIPPED_TASK_PROMPTS = [
    # '{src_lang}: "{src_word}" - There is nothing: "{tgt_word}',
    # '{src_lang}: "{src_word}" - Nothing nothing: "{tgt_word}',
    '{src_lang}: "{src_word}" - Nothing: "{tgt_word}',
    # 'Translate "{src_word}" into Nothing: "',
    # 'Translate the {src_lang} word "{src_word}" to Nothing: "',
    # 'From {src_lang}: "{src_word}" to Nothing: "',
    # 'Provide the translation of "{src_word}" from {src_lang} to Nothing: "',
    # 'Q: What is "{src_word}" in Nothing? A: "',
    # 'Q: What is the Nothing translation "{src_word}"? A: "',
    # 'Q: How do you say "{src_word}" in Nothing? A: "',
    # 'Q: What is "{src_word}" translated into Nothing? A: "'
]

def get_random_idx():
    return random.randint(0, len(TRANSLATION_PROMPTS) - 1)

# def get_random_item(in_list:list):
#     idx = random.randint(0,len(in_list)-1)
#     return in_list[idx]

# def sample(df, ind, k=5, tokenizer=None, lang1='fr', lang2='de'):
#     df = df.reset_index(drop=True)
#     temp = df[df.index!=ind]
#     sample = pd.concat([temp.sample(k-1), df[df.index==ind]], axis=0)
#     prompt = ""
#     for idx, (df_idx, row) in enumerate(sample.iterrows()):
#         if idx < k-1:
#             prompt += f'{LANG2NAME[lang1]}: "{row[lang1]}" - {LANG2NAME[lang2]}: "{row[lang2]}"\n'
#         else:
#             # prompt += f'{LANG2NAME[lang1]}: "{row[lang1]}" - {LANG2NAME[lang2]}: "'
#             prompt += get_random_item(TRANSLATION_PROMPTS).format(src_lang=LANG2NAME[lang1], tgt_lang=LANG2NAME[lang2], src_word=row[lang1], tgt_word=row[lang2])
#             in_token_str = row[lang1]
#             out_token_str = row[lang2]
#             out_token_id = process_tokens(out_token_str, tokenizer, lang2)
#             # latent_token_str = row[lang_latent]
#             # latent_token_id = process_tokens(latent_token_str, tokenizer, 'en')
#             # intersection = set(out_token_id).intersection(set(latent_token_id))
#             # if len(out_token_id) == 0 or len(latent_token_id) == 0:
#                 # yield None
#             # if lang2 != 'en' and len(intersection) > 0:
#                 # yield None
#             yield {'prompt': prompt, 
#                 'out_token_id': out_token_id, 
#                 'out_token_str': out_token_str,
#                 # 'latent_token_id': latent_token_id, 
#                 # 'latent_token_str': latent_token_str, 
#                 'in_token_str': in_token_str}

# def flipped_task_sample(df, ind, k=5, tokenizer=None, lang1='fr', lang2='de'):
#     df = df.reset_index(drop=True)
#     temp = df[df.index!=ind]
#     sample = pd.concat([temp.sample(k-1), df[df.index==ind]], axis=0)
#     prompt = ""
#     for idx, (df_idx, row) in enumerate(sample.iterrows()):
#         if idx < k-1:
#             prompt += f'{LANG2NAME[lang1]}: "{row[lang1]}" - {LANG2NAME[lang2]}: "{row[lang2]}"\n'
#         else:
#             prompt += get_random_item(FLIPPED_TASK_PROMPTS).format(src_lang=LANG2NAME[lang1], src_word=row[lang1], tgt_word=row[lang2])
#             in_token_str = row[lang1]
#             out_token_str = row[lang2]
#             out_token_id = process_tokens(out_token_str, tokenizer, lang2)
#             # latent_token_str = row[lang_latent]
#             # latent_token_id = process_tokens(latent_token_str, tokenizer, 'en')
#             # intersection = set(out_token_id).intersection(set(latent_token_id))
#             # if len(out_token_id) == 0 or len(latent_token_id) == 0:
#                 # yield None
#             # if lang2 != 'en' and len(intersection) > 0:
#                 # yield None
#             yield {'prompt': prompt, 
#                 'out_token_id': out_token_id, 
#                 'out_token_str': out_token_str,
#                 # 'latent_token_id': latent_token_id, 
#                 # 'latent_token_str': latent_token_str, 
#                 'in_token_str': in_token_str}


# def _get_preprocess_df_trans(df_trans, tokenizer, src_lang, tgt_lang):
#     dataset = []
#     for ind in tqdm(range(len(df_trans))):
#         d = next(sample(df_trans, ind, k=1, tokenizer=tokenizer, lang1=src_lang, lang2=tgt_lang))
#         if d is None:
#             continue
#         dataset.append(d)
#     return dataset

# def _get_flipped_preprocess_df_trans(df_trans, tokenizer, src_lang, tgt_lang, flipped_type="task"):
#     dataset = []
#     for ind in tqdm(range(len(df_trans))):
#         if flipped_type == "task":
#             d = next(flipped_task_sample(df_trans, ind, k=1, tokenizer=tokenizer, lang1=src_lang, lang2=tgt_lang))
#         else:
#             raise NotImplementedError
#         if d is None:
#             continue
#         dataset.append(d)
#     return dataset

def unified_sample(df, ind, k=5, tokenizer=None, lang1='fr', lang2='de', flipped_type='task'):
    df = df.reset_index(drop=True)
    temp = df[df.index != ind]
    sample = pd.concat([temp.sample(k-1), df[df.index == ind]], axis=0)

    prompt_normal = ""
    prompt_flipped = ""
    for idx, (df_idx, row) in enumerate(sample.iterrows()):
        prompt_idx = get_random_idx()
        normal_translation_prompt = TRANSLATION_PROMPTS[prompt_idx]
        if flipped_type == "task":
            flipped_prompt = FLIPPED_TASK_PROMPTS[prompt_idx]
        else:
            print(flipped_type)
            raise NotImplementedError
        if idx < k-1:
            prompt_normal += f'{normal_translation_prompt.format(src_lang=LANG2NAME[lang1], tgt_lang=LANG2NAME[lang2], src_word=row[lang1])}{row[lang2]}"\n'
            prompt_flipped += f'{flipped_prompt.format(src_lang=LANG2NAME[lang1], tgt_lang=LANG2NAME[lang2], src_word=row[lang1])}{row[lang2]}"\n'
        else:
            # Normal Prompt
            prompt_normal += normal_translation_prompt.format(
                src_lang=LANG2NAME[lang1], tgt_lang=LANG2NAME[lang2], src_word=row[lang1], tgt_word=row[lang2]
            )
            # Flipped Prompt
            prompt_flipped += flipped_prompt.format(
                src_lang=LANG2NAME[lang1], tgt_lang=LANG2NAME[lang2], src_word=row[lang1], tgt_word=row[lang2]
            )

            in_token_str = row[lang1]
            out_token_str = row[lang2]

            in_token_id = tokenizer.encode(in_token_str, add_special_tokens=False)
            # Process the tokens for both normal and flipped cases
            out_token_id = process_tokens(out_token_str, tokenizer, lang2)
            # For now, flipped and normal tasks share the same token info
            # If needed, you can modify the token processing here for flipped cases as required.

            yield {
                'normal_prompt': prompt_normal, 
                'flipped_prompt': prompt_flipped, 
                'out_token_id': out_token_id, 
                'out_token_str': out_token_str,
                'in_token_id': in_token_id,
                'in_token_str': in_token_str,
                'template_idx': prompt_idx,
            }

def _get_unified_preprocess_df_trans(df_trans, tokenizer, src_lang, tgt_lang, flipped_type):
    dataset_normal = []
    dataset_flipped = []
    
    # Iterate through each index to collect both normal and flipped samples
    for ind in tqdm(range(len(df_trans))):
        d = next(unified_sample(df_trans, ind, k=1, tokenizer=tokenizer, lang1=src_lang, lang2=tgt_lang, flipped_type=flipped_type))
        if d is None:
            continue
        
        # Collect normal dataset and flipped dataset separately
        dataset_normal.append({
            'prompt': d['normal_prompt'], 
            'out_token_id': d['out_token_id'], 
            'out_token_str': d['out_token_str'],
            'in_token_id': d['in_token_id'], 
            'in_token_str': d['in_token_str'],
            'template_idx': d['template_idx'],
        })
        
        dataset_flipped.append({
            'prompt': d['flipped_prompt'], 
            'out_token_id': d['out_token_id'], 
            'out_token_str': d['out_token_str'],
            'in_token_id': d['in_token_id'], 
            'in_token_str': d['in_token_str'],
            'template_idx': d['template_idx'],
        })
    
    return dataset_normal, dataset_flipped


class TranslationDataset:
    def __init__(
        self,
        tokenizer=None,
        tokenizer_path=None,
        src_lang=None,
        tgt_lang=None,
        single_token_only=False,
        multi_token_only=False,
        prepend_bos=True,
        preprocess_df_trans=None,
        flipped_type=None
    ):
        """
        ioi_prompts_for_word_idxs:
            if you want to use a different set of prompts to get the word indices, you can pass it here
            (example use case: making a ABCA dataset)
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            raise NotImplementedError
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.id2voc = {id:voc for voc, id in self.tokenizer.get_vocab().items()}

        self.df_trans = _construct_translation_prompts(
            tokenizer=self.tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            single_token_only=single_token_only,
            multi_token_only=multi_token_only
        )
        
        # self.df_trans = _construct_new_translation_prompts(
        #     src_lang=src_lang,
        #     tgt_lang=tgt_lang,
        # )
        
        # self.N = len(self.df_trans)

        if preprocess_df_trans is not None:
            self.preprocess_df_trans = preprocess_df_trans
            # if preprocess_flipped_df_trans is not None:
            #     self.preprocess_flipped_df_trans = preprocess_flipped_df_trans
        else:
            self.preprocess_df_trans, self.preprocess_flipped_df_trans = _get_unified_preprocess_df_trans(self.df_trans, self.tokenizer, src_lang, tgt_lang, flipped_type)
        
        self.N = len(self.preprocess_df_trans)
        
        all_ids = [prompt["template_idx"] for prompt in self.preprocess_df_trans]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])

        small_groups = []
        for group in self.groups:
            if len(group) < 5:
                small_groups.append(len(group))
        if len(small_groups) > 0:
            warnings.warn(
                f"Some groups have less than 5 prompts, they have lengths {small_groups}"
            )
        
        self.sentences = [
            prompt["prompt"] for prompt in self.preprocess_df_trans
        ]  # a list of strings. Renamed as this should NOT be forward passed

        # print(self.ioi_prompts, "that's that")
        texts = [prompt["prompt"] for prompt in self.preprocess_df_trans]
        tokenized_text_input_dict = self.tokenizer(texts, padding=True, add_special_tokens=prepend_bos, return_tensors="pt")
        self.toks = tokenized_text_input_dict.input_ids
        self.in_tok_ids = [item["in_token_id"] for item in self.preprocess_df_trans]
        self.out_tok_ids = [item["out_token_id"] for item in self.preprocess_df_trans]
        self.attn_mask = tokenized_text_input_dict.attention_mask

        self.word_idx = get_idx_dict(
            toks=self.toks,
            attn_mask=self.attn_mask,
            tokenizer=tokenizer
        )

        self.translation_tokenIDs = [item['out_token_id'] for item in self.preprocess_df_trans]
        
        self.prepend_bos = prepend_bos

        
        self.max_len = max(
            [
                len(self.tokenizer(prompt["prompt"]).input_ids)
                for prompt in self.preprocess_df_trans
            ]
        )

        self.tokenized_prompts = []

        for i in range(self.N):
            self.tokenized_prompts.append(
                "|".join([self.tokenizer.decode(tok) for tok in self.toks[i]])
            )


    def gen_filpped_dataset(self):
        flipped_translation_dataset = TranslationDataset(
            tokenizer=self.tokenizer,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            single_token_only = True,
            multi_token_only = False,
            preprocess_df_trans=self.preprocess_flipped_df_trans
        )
        return flipped_translation_dataset

    def __getitem__(self, key):
        sliced_prompts = self.preprocess_df_trans[key]
        sliced_dataset = TranslationDataset(
            tokenizer=self.tokenizer,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            single_token_only = True,
            multi_token_only = False,
            preprocess_df_trans=sliced_prompts,
        )
        return sliced_dataset

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __len__(self):
        return self.N

    def tokenized_prompts(self):
        return self.toks