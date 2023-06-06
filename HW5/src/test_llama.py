import pdb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

model_path = r'/home/incoming/llm/llama/llama-30b'
peft_path = r'/home/incoming/llm/llama/SuperCOT-LoRA/30b/gpu/cutoff-1024'

template = '''### Instruction:
{}

### Input:
{}

### Response:
'''

quant_config = {
    'load_in_4bit': True,
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_compute_dtype': torch.bfloat16
}

nf4_config = BitsAndBytesConfig(**quant_config)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model_peft = PeftModel.from_pretrained(model_nf4, peft_path)

ppl = pipeline('text-generation', model=model_peft, tokenizer=tokenizer)
tg_config = {
    'max_length': 300,
    'do_sample': False,
    'eos_token_id': tokenizer.eos_token_id
    # temperature, top_k, top_p, num_beams, num_return_sequences
}


def get_result(instruction, inp):
    text = template.format(instruction, inp)
    seqs = ppl(text, **tg_config)
    for seq in seqs:
        print(seq['generated_text'])
    return seqs

if __name__ == '__main__':
    # 将 instruction 和 input 中的 \n\n 转换为 \n
    instruction = 'Transform the following text into json format with keys:\n+ Object: the object to operate on;\n+ Goal: the goal to be achieved by the operation;\n+ Method: the method to achieve the goal.'
    inp = 'SuperCoT is a LoRA trained with the aim of making LLaMa follow prompts for Langchain better by, infusing chain-of-thought datasets, code explanations and instructions, snippets, logical deductions and Alpaca GPT-4 prompts.'
    seqs = get_result(instruction, inp)

    pdb.set_trace()