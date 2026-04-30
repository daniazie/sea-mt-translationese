from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling, DataCollatorForVisionLanguageModeling
from trl.trainer.dpo_trainer import DataCollatorForPreference, DataCollatorForVisionPreference
from dataclasses import dataclass, asdict

@dataclass
class FormatContentForVLM:
    text: str
    type: str = "text"

@dataclass
class FormatInputsForChat:
    role: str
    content: str | list[FormatContentForVLM]

def get_data_collator(train_mode, is_vl, processor, **kwargs) -> DataCollatorMixin:
    if train_mode == 'sft':
        if is_vl:
            return DataCollatorForVisionLanguageModeling(processor=processor, **kwargs)
        return DataCollatorForLanguageModeling(**kwargs)
    elif train_mode == 'dpo':
        if is_vl:
            return DataCollatorForVisionPreference(processor=processor, **kwargs)
        return DataCollatorForPreference(**kwargs)

def system_prompt_supported(tokenizer: PreTrainedTokenizerBase | ProcessorMixin):
    try: 
        _ = tokenizer.apply_chat_template([{"role": "system", "content": "Lorem ipsum"}])
        return True
    except:
        return False

def format_prompt_completion(examples, tokenizer, is_vl, use_messages_format):
    srcs = [example for example in examples['src']]
    refs = [example for example in examples['ref']]
    src_langs = [example for example in examples['src_lang']]
    tgt_langs = [example for example in examples['tgt_lang']] 

    prompts = []
    completions = []
    instruction = "Translate the given sentence from {src_lang} to {tgt_lang}.\n{src_lang}: {src}"
    for src, ref, src_lang, tgt_lang in zip(srcs, refs, src_langs, tgt_langs):
        user_prompt = instruction.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)
        if is_vl:
            user_prompt = [asdict(FormatContentForVLM(text=user_prompt))]
        if use_messages_format:
            prompt = [asdict(FormatInputsForChat(role='user', content=user_prompt))]
            if is_vl:
                ref = [asdict(FormatContentForVLM(text=ref))]
            completion = [asdict(FormatInputsForChat(role='assistant', content=ref))]
        else:
            prompt = user_prompt
            completion = ref
        
        prompts.append(prompt)        
        completions.append(completion)

    return {'prompt': prompts, 'completion': completions}

def format_paired_dataset(examples, chosen_key, rejected_key, is_vl, use_messages_format, src_lang, tgt_lang):
    srcs = [example for example in examples['src']]
    chosen_texts = [example for example in examples[chosen_key]]
    rejected_texts = [example for example in examples[rejected_key]]

    prompts = []
    completions_chosen = []
    completions_rejected = []
    instruction = "Translate the given sentence from {src_lang} to {tgt_lang}.\n{src_lang}: {src}\n\n{tgt_lang}: "
    for src, chosen, rejected in zip(srcs, chosen_texts, rejected_texts):
        user_prompt = instruction.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)
        if is_vl:
            user_prompt = [asdict(FormatContentForVLM(text=user_prompt))]
        if use_messages_format:
            prompt = [asdict(FormatInputsForChat(role='user', content=user_prompt))]
            if is_vl:
                chosen = [asdict(FormatContentForVLM(text=chosen))]
                rejected = [asdict(FormatContentForVLM(text=rejected))]
            chosen = [asdict(FormatInputsForChat(role='assistant', content=chosen))]
            rejected = [asdict(FormatInputsForChat(role='assistant', content=rejected))]
        else:
            prompt = user_prompt
        
        prompts.append(prompt)        
        completions_chosen.append(chosen)
        completions_rejected.append(rejected)

    return {'prompt': prompts, 'chosen': completions_chosen, "rejected": completions_rejected}

def format_messages(examples, tokenizer, is_vl):
    srcs = [example for example in examples['src']]
    refs = [example for example in examples['ref']]
    src_langs = [example for example in examples['src_lang']]
    tgt_langs = [example for example in examples['tgt_lang']] 

    messages = []
    sys_prompt = "Translate the given sentence from {src_lang} to {tgt_lang}."
    user_prompt = "{src_lang}: {src}\n\n{tgt_lang}: "
    for src, ref, src_lang, tgt_lang in zip(srcs, refs, src_langs, tgt_langs):
        if not system_prompt_supported(tokenizer):
            instruction = sys_prompt + '\n\n' + user_prompt
            instruction = instruction.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)

            if is_vl:
                instruction = [asdict(FormatContentForVLM(text=instruction))]
                ref = [asdict(FormatContentForVLM(text=ref))]
            message = [
                asdict(FormatInputsForChat(role='user', content=instruction)),
                asdict(FormatInputsForChat(role='assistant', content=ref))
            ]
            messages.append(message)

        else:
            system = sys_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang)
            user = user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)
            if is_vl:
                user = [asdict(FormatContentForVLM(text=user))]
                ref = [asdict(FormatContentForVLM(text=ref))]
            message = [
                asdict(FormatInputsForChat(role='system', content=system)),
                asdict(FormatInputsForChat(role='user', content=user)),
                asdict(FormatInputsForChat(role='assistant', content=ref))
            ]
            
            messages.append(message)

    return {'messages': messages}

def preprocess_dataset(example, tokenizer: PreTrainedTokenizerBase):
    if example.get("prompt") and example.get("completion"):
        return {
            "prompt": tokenizer.apply_chat_template(example['prompt'], tokenize=False, add_generation_prompt=True),
            "completion": tokenizer.apply_chat_template(example['completion'], tokenize=False)
        }
    return {
        "text": tokenizer.apply_chat_template(
            example['messages'],
        )
    }