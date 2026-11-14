"""Independent Transformers Qwen3 oracle with tiny deterministic weights.

Fixtures test native model mechanics, not language quality of these random models.
Requires torch, transformers==4.53.3, safetensors, and tokenizers.
"""
import json
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3Model,Qwen3ForCausalLM
from tokenizers import Tokenizer,AddedToken
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from generate import upstream_module,ROOT

torch.manual_seed(91)
torch.set_num_threads(1)
root=ROOT/"core/fixtures/pipeline"
root.mkdir(parents=True,exist_ok=True)
vocab={s:i for i,s in enumerate(["[UNK]","<|im_start|>","<|im_end|>","<|endoftext|>","<|memory|>",
    "system","user","assistant","Memory","<think>","</think>","alpha","beta","gamma","delta","Which","value","?","Use","memory","."])}
vocab.update({f"t{i}":i for i in range(len(vocab),64)})
tokenizer=Tokenizer(WordLevel(vocab,unk_token="[UNK]"))
tokenizer.pre_tokenizer=Whitespace()
tokenizer.add_special_tokens([AddedToken(x,special=True) for x in ["<|im_start|>","<|im_end|>","<|endoftext|>","<|memory|>","<think>","</think>"]])

def config(hidden,heads):
    return Qwen3Config(vocab_size=64,hidden_size=hidden,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=heads,num_key_value_heads=1,head_dim=4,max_position_embeddings=256,
        attention_bias=False,hidden_act="silu",use_sliding_window=False,tie_word_embeddings=False,
        bos_token_id=3,eos_token_id=2)

encoder=Qwen3Model(config(8,3)).eval()
decoder=Qwen3ForCausalLM(config(12,4)).eval()
for name,model in [("encoder",encoder),("decoder",decoder)]:
    directory=root/name
    model.save_pretrained(directory,safe_serialization=True)
    tokenizer.save(str(directory/"tokenizer.json"))
    cfg=json.loads((directory/"config.json").read_text())
    cfg["max_window_layers"]=2
    (directory/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")

memory=upstream_module().MemoryModule.from_pretrained(ROOT/"core/fixtures/learned").eval()
documents=["alpha beta","gamma delta"]
question="Which value ?"
system="Use memory ."
def encode(text):
    ids=torch.tensor([tokenizer.encode(text).ids])
    h=encoder(ids).last_hidden_state[:,-1].float()
    return torch.nn.functional.normalize(h,dim=-1).half().float()

with torch.no_grad():
    docs=torch.cat([encode(x) for x in documents])
    query=encode(question)
    memories=memory(docs[None],None,query)
    prompt=f"<|im_start|>system\n{system}\nMemory{'<|memory|>'*3}Memory<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids=torch.tensor([tokenizer.encode(prompt).ids])
    embeds=decoder.get_input_embeddings()(ids)
    embeds[ids==vocab["<|memory|>"]]=memories.reshape(-1,12)
    logits=decoder(inputs_embeds=embeds).logits
    out=decoder.generate(inputs_embeds=embeds,max_new_tokens=5,do_sample=False,eos_token_id=[2,3],pad_token_id=3)
    # Distinct head projection width catches hidden_size != heads * head_dim.
    synthetic=torch.randn(1,4,12)
    h=decoder.model(inputs_embeds=synthetic,use_cache=True)
    next_h=decoder.model(torch.tensor([[11]]),past_key_values=h.past_key_values,use_cache=True)
    receipt=dict(documents=documents,question=question,system=system,document_embeddings=docs.tolist(),
        query=query.tolist(),expected=tokenizer.decode(out[0].tolist(),skip_special_tokens=True),
        input_embeds=synthetic.tolist(),hidden=h.last_hidden_state.tolist(),next_hidden=next_h.last_hidden_state.tolist(),
        next_token=11,first_logits=logits[:,-1].tolist(),torch=torch.__version__)
    (root/"golden.json").write_text(json.dumps(receipt,separators=(",",":"))+"\n")
    print("pipeline oracle",receipt["expected"],flush=True)
