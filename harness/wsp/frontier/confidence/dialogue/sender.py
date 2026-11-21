"""Run the pinned authors' FP32 EmoBERTa classifier on current utterances."""
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification

from dialogue_data import ensure


class Sender:
    def __init__(self, cache):
        torch.set_num_threads(4)
        cfg = RobertaConfig.from_json_file(str(ensure(cache,"emoberta-config.json")))
        self.anger = next(i for i,label in cfg.id2label.items() if label == "anger")
        if self.anger != 3 or len(cfg.id2label) != 7:
            raise ValueError("source class mapping differs")
        cfg._attn_implementation = "eager"
        checkpoint = torch.load(ensure(cache,"emoberta-base.bin"),map_location="cpu",weights_only=True)
        # Old Transformers serialized this fixed buffer. Modern RoBERTa regenerates it.
        positions = checkpoint.pop("roberta.embeddings.position_ids")
        if not torch.equal(positions, torch.arange(cfg.max_position_embeddings)[None,:]):
            raise ValueError("source position buffer changed")
        self.model = RobertaForSequenceClassification(cfg)
        self.model.load_state_dict(checkpoint,strict=True)
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(ensure(cache,"emoberta-tokenizer.json")))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=96)
        self.pad = cfg.pad_token_id

    @torch.inference_mode()
    def margins(self, texts):
        tokens = [v.ids for v in self.tokenizer.encode_batch(list(texts))]
        order = sorted(range(len(tokens)),key=lambda j:len(tokens[j]))
        result = np.empty(len(order),dtype=np.float64)
        for start in range(0,len(order),16):
            ids = order[start:start+16]
            width = len(tokens[ids[-1]])
            inputs = torch.full((len(ids),width),self.pad,dtype=torch.long)
            mask = torch.zeros_like(inputs)
            for i,j in enumerate(ids):
                length = len(tokens[j])
                inputs[i,:length] = torch.as_tensor(tokens[j],dtype=torch.long)
                mask[i,:length] = 1
            logits = self.model(input_ids=inputs,attention_mask=mask).logits.double().numpy()
            other = np.delete(logits,self.anger,axis=1)
            result[ids] = logits[:,self.anger] - np.logaddexp.reduce(other,axis=1)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite native sender margin")
        return result
