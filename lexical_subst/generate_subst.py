import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
from tqdm.auto import tqdm

tqdm.pandas()


def mask_before_target(idxs, line):
    start_id = int(idxs.split(',')[0].split('-')[0].strip())
    return line[:start_id] + '[MASK] а также ' + line[start_id:]


def mask_after_target(idxs, line):
    end_id = int(idxs.split(',')[0].split('-')[1].strip())
    return line[:end_id + 1] + ' а также [MASK]' + line[end_id + 1:]


def load_models(modelname):
    tokenizer = BertTokenizer.from_pretrained(modelname)
    model = BertForMaskedLM.from_pretrained(modelname)
    return tokenizer, model


def predict_masked_sent(tokenizer, model, text, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]" % text
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    out = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        out.append((token_weight, predicted_token))
    return out


# def extract_ling_feats(idx, text):
#     start_id = int(idx.split('-')[0].strip())
#     end_id = int(idx.split('-')[1].strip())
#     processed = nlp(text)
#     case = morph.parse(text[start_id:end_id+1])[0].tag.case
#     number = morph.parse(text[start_id:end_id+1])[0].tag.number
#     dep=''
#     for token in processed.iter_tokens():
#         if start_id == token.start_char:
#             dep = token.words[0].deprel
#             break
#     return case, number, dep


def generate(path, modelname):
    tokenizer, model = load_models(modelname)
    df = pd.read_csv(path, sep='\t')
    df['masked_before_target'] = df[['positions', 'context']].apply(lambda x: mask_before_target(*x), axis=1)
    # df['masked_after_target'] = df[['positions', 'context']].apply(lambda x: mask_after_target(*x), axis=1)
    df['before_subst_prob'] = df['masked_before_target'].progress_apply(lambda x: predict_masked_sent(tokenizer, model, x,
                                                                                                 top_k=150))
    # df['after_subst'] = df['masked_after_target'].progress_apply(lambda x: predict_masked_sent(tokenizer, model, x,
    #                                                                                            top_k=150))
    return df
