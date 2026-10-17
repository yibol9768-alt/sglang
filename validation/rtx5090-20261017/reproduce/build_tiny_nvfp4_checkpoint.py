"""A local synthetic two-layer Qwen3 MoE checkpoint for loader regression."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig

ROOT=Path(__file__).resolve().parent
DEST=ROOT/"models/tiny-qwen3-moe-nvfp4"

def main():
    torch.manual_seed(5090)
    config=Qwen3MoeConfig(
        vocab_size=256,hidden_size=256,intermediate_size=512,
        moe_intermediate_size=256,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=2,head_dim=64,num_experts=4,num_experts_per_tok=2,
        max_position_embeddings=512,tie_word_embeddings=False,
        eos_token_id=2,bos_token_id=1,
    )
    config.architectures=["Qwen3MoeForCausalLM"]
    config.dtype="bfloat16"
    quant={"num_bits":4,"type":"float","symmetric":True,"group_size":16,
           "strategy":"tensor_group","dynamic":False,"observer":"minmax"}
    activation=dict(quant,dynamic="local")
    config.quantization_config={
        "quant_method":"compressed-tensors","format":"nvfp4-pack-quantized",
        "quantization_status":"compressed","kv_cache_scheme":None,
        "ignore":["re:.*self_attn.*","re:.*mlp.gate$","lm_head"],
        "config_groups":{"group_0":{"targets":["Linear"],"weights":quant,
          "input_activations":activation,"output_activations":None}},
    }
    weights={}
    def normal(name,shape):weights[name]=(torch.randn(shape)*0.02).bfloat16()
    normal("model.embed_tokens.weight",(256,256))
    normal("lm_head.weight",(256,256))
    weights["model.norm.weight"]=torch.ones(256,dtype=torch.bfloat16)
    for layer in range(2):
        p=f"model.layers.{layer}."
        for name,size in [("q_proj",256),("k_proj",128),("v_proj",128),("o_proj",256)]:
            normal(p+f"self_attn.{name}.weight",(size,256))
        for name in ["q_norm","k_norm"]:
            weights[p+f"self_attn.{name}.weight"]=torch.ones(64,dtype=torch.bfloat16)
        for name in ["input_layernorm","post_attention_layernorm"]:
            weights[p+f"{name}.weight"]=torch.ones(256,dtype=torch.bfloat16)
        normal(p+"mlp.gate.weight",(4,256))
        for expert in range(4):
            for name in ["gate_proj","up_proj","down_proj"]:
                prefix=p+f"mlp.experts.{expert}.{name}."
                weights[prefix+"weight_packed"]=torch.randint(0,256,(256,128),dtype=torch.uint8)
                weights[prefix+"weight_scale"]=torch.ones((256,16),dtype=torch.float8_e4m3fn)
                weights[prefix+"weight_global_scale"]=torch.tensor(128.0)
                weights[prefix+"input_global_scale"]=torch.tensor(128.0)
    DEST.mkdir(parents=True,exist_ok=True)
    config.save_pretrained(DEST)
    save_file(weights,DEST/"model.safetensors",metadata={"format":"pt"})
    print(json.dumps({"model_path":str(DEST),"checkpoint_bytes":sum(p.numel()*p.element_size() for p in weights.values()),"tensors":len(weights)}))

if __name__=="__main__":main()
