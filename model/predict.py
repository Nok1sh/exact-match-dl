import torch
from model.parameters import Params


def predict(model, label, text):

    model.eval()

    inputs = Params.TOKENIZER(
        label,
        text,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    ).to(Params.DEVICE)

    offsets = inputs.pop("offset_mapping")
    inputs = {k: v.to(Params.DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)

    scores = start_probs[:, 1:].max(dim=-1).values * end_probs[:, 1:].max(dim=-1).values
    window_ind = torch.argmax(scores).item()
    
    start_token_ind = torch.argmax(start_logits[window_ind]).item()
    end_token_ind = torch.argmax(end_logits[window_ind]).item()

    start_ind = offsets[window_ind][start_token_ind][0].item()
    end_ind = offsets[window_ind][end_token_ind][1].item()

    answer = text[start_ind:end_ind]

    return answer, start_ind, end_ind