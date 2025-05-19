from transformers import LlamaForCausalLM

class GetAttnMapLM(LlamaForCausalLM):
    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        if input_ids.shape[1] == 1:
            kwargs["output_attentions"] = True
        return super().forward(**kwargs)
