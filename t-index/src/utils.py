from datasets import Dataset
import json

def format_messages(examples, is_vl):
    srcs = [example for example in examples['src']]
    refs = [example for example in examples['ref']]
    mts = [example for example in examples['mt']]
    src_langs = [example for example in examples['src_lang']]
    tgt_langs = [example for example in examples['tgt_lang']] 

    prompts = []
    completions_chosen = []
    completions_rejected = []
    sys_prompt = "Translate the given sentence from {src_lang} to {tgt_lang}."
    user_prompt = "{src_lang}: {src}\n\n{tgt_lang}: "
    for src, ref, mt, src_lang, tgt_lang in zip(srcs, refs, mts, src_langs, tgt_langs):
        prompt = [
            {
                "role": "system", "content": [{"type": "text", "text": sys_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang)}] if is_vl else sys_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang)
            },
            {
                "role": "user", "content": [{"type": "text", "text": user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)}] if is_vl else user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)
            }
        ]

        chosen = [
            {
            "role": "assistant", "content": [{"type": "text", "text": ref}] if is_vl else ref
            }
        ]
        
        rejected = [
            {
            "role": "assistant", "content": [{"type": "text", "text": mt}] if is_vl else mt
            }
        ]
        
        prompts.append(prompt)
        completions_chosen.append(chosen)
        completions_rejected.append(rejected)

    return {"prompt": prompt, "chosen": completions_chosen, "rejected": completions_rejected}