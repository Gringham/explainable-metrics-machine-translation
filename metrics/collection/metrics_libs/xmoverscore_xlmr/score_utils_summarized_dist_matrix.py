from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import string
import tqdm
from pyemd import emd
from pyemd.emd import emd_with_flow


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        x_encoded_layers = model(input_ids = x, token_type_ids = None, attention_mask = attention_mask)[2]
    return x_encoded_layers


def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="<pad>", device='cuda:0'):
    # Here we need to truncate to the max length of bert inside of the representation
    tokens = [["<s>"]+tokenize(a)[:510]+["</s>"] for a in arr]
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens


def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device='cuda:0'):

    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                              tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                              idf_dict,
                                                              device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i + batch_size],
                                          attention_mask=mask[i:i + batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def slide_window(input_, w=3, o=2):
    if input_.size - w + 1 <= 0:
        w = input_.size
    sh = (input_.size - w + 1, w)
    st = input_.strides * 2
    view = np.lib.stride_tricks.as_strided(input_, strides=st, shape=sh)[0::o]
    return view.copy().tolist()


def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)


def load_ngram(ids, embedding, idf, n, o, device='cuda:0'):
    new_a = []
    new_idf = []

    slide_wins = slide_window(np.array(ids), w=n, o=o)
    for slide_win in slide_wins:
        new_idf.append(idf[slide_win].sum().item())
        scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(device)
        tmp = (scale * embedding[slide_win]).sum(0)
        new_a.append(tmp)
    new_a = torch.stack(new_a, 0).to(device)
    return new_a, new_idf


from collections import defaultdict


def cross_lingual_mapping(mapping, embedding, projection, bias):
    batch_size = embedding.shape[0]
    n_tokens = embedding.shape[1]

    if mapping == 'CLP':
        embedding = torch.matmul(embedding, projection)
    if mapping == 'UMD':
        embedding = embedding - (embedding * bias).sum(2, keepdim=True) * bias.repeat(batch_size, n_tokens, 1)
    return embedding


def lm_perplexity(model, hyps, tokenizer, batch_size=1, device='cuda:0'):
    preds = []
    model.eval()
    for batch_start in range(0, len(hyps), batch_size):
        batch_hyps = hyps[batch_start:batch_start + batch_size]

        tokenize_input = tokenizer.tokenize(batch_hyps[0])

        if len(tokenize_input) <= 1:
            preds.append(0)
        else:
            if len(tokenize_input) > 1024:
                tokenize_input = tokenize_input[:1024]

            arr = tokenizer.convert_tokens_to_ids(tokenize_input)
            input_ids = torch.tensor([arr])
            input_ids = input_ids.to(device=device)
            score = model(input_ids, labels=input_ids)[0]
            preds.append(-score.item())
    return preds


def word_mover_score(mapping, projection, bias, model, tokenizer, src, hyps, n_gram=1, batch_size=256, device='cuda:0', layer = 21):
    idf_dict_src = defaultdict(lambda: 1.)
    idf_dict_hyp = defaultdict(lambda: 1.)

    preds = []
    token_relevances = []
    for batch_start in tqdm.tqdm(range(0, len(src), batch_size)):
        batch_src = src[batch_start:batch_start + batch_size]
        batch_hyps = hyps[batch_start:batch_start + batch_size]

        src_embedding, src_lens, src_masks, src_idf, src_tokens = get_bert_embedding(batch_src, model, tokenizer,
                                                                                     idf_dict_src,
                                                                                     device=device)
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(batch_hyps, model, tokenizer,
                                                                                     idf_dict_hyp,
                                                                                     device=device)

        src_embedding = src_embedding[layer]
        hyp_embedding = hyp_embedding[layer]

        #src_embedding = cross_lingual_mapping(mapping, src_embedding, projection, bias[0])

        batch_size = len(src_embedding)

        for i in range(batch_size):
            src_ids = [k for k, w in enumerate(src_tokens[i])]
            hyp_ids = [k for k, w in enumerate(hyp_tokens[i])]
            #src_considered = collapse_list(src_tokens[i])
            #hyp_considered = collapse_list(hyp_tokens[i])
            #src_considered = [w for k, w in enumerate(src_considered) if w not in set(string.punctuation)]
            #hyp_considered = [w for k, w in enumerate(hyp_considered) if w not in set(string.punctuation)]
            src_considered = src_tokens[i]
            hyp_considered = hyp_tokens[i]

            def collapse_list(src, hyp, dist):
                l = src + hyp
                src = [l[1].replace('▁','')]
                hyp = []
                to = 'SRC'
                inb = 1
                # remove beginning sep
                dist = np.delete(dist, 0, 0)
                dist = np.delete(dist, 0, 1)
                y = 1
                for x in range(2, len(l)):
                    if l[x] == '<s>' or l[x] == '</s>':
                        dist = np.delete(dist, y, 0)
                        dist = np.delete(dist, y, 1)
                        if l[x] == '</s>':
                            to = 'HYP'
                        if len(l) - 1 == x:
                            dist[-1] /= inb
                            dist[:, -1] /= inb

                    elif l[x][0] != '▁' and l[x][0] != 'Ġ':
                        if to == 'SRC':
                            src[-1] += l[x]
                        else:
                            hyp[-1] += l[x]
                        dist[y-1] += dist[y]
                        dist[:,y-1] += dist[:,y]
                        dist = np.delete(dist, y, 0)
                        dist = np.delete(dist, y, 1)

                        inb += 1
                        if len(l) - 1 == x:
                            dist[-1] /= inb
                            dist[:,-1] /= inb
                    else:
                        # Remove the underscore
                        dist[y-1] /= inb
                        dist[:, y - 1] /= inb
                        inb = 1
                        if to == 'SRC':
                            src.append(l[x][1:])
                        else:
                            hyp.append(l[x][1:])
                        y+=1
                return src, hyp, dist



            src_grams = [','.join(window) for window in slide_window(np.array(src_considered), w=n_gram, o=1)]
            #hyp_grams = [','.join(window) for window in slide_window(np.array(hyp_considered), w=n_gram, o=1)]

            src_embedding_i, src_idf_i = load_ngram(src_ids, src_embedding[i], src_idf[i], n_gram, 1)
            hyp_embedding_i, hyp_idf_i = load_ngram(hyp_ids, hyp_embedding[i], hyp_idf[i], n_gram, 1)

            embeddings = torch.cat([src_embedding_i, hyp_embedding_i], 0)
            embeddings.div_(torch.norm(embeddings, dim=-1).unsqueeze(-1) + 1e-30)
            distance_matrix = pairwise_distances(embeddings, embeddings)

            dist_mt = distance_matrix.double().cpu().numpy()

            src_l, hyp_l, dist_mt = collapse_list(src_considered, hyp_considered, dist_mt)
            #c1 = np.zeros(len(src_idf_i) + len(hyp_idf_i))
            #c2 = np.zeros_like(c1)

            #c1[:len(src_idf_i)] = src_idf_i
            #c2[-len(hyp_idf_i):] = hyp_idf_i
            c1 = np.array([1]*len(src_l)+[0]*len(hyp_l))
            c2 = np.array([0] * len(src_l) + [1] * len(hyp_l))
            #inter_sentence = np.min(dist_mt[len(src_grams):,:len(src_grams)],axis=1)
            inter_sentence = np.min(dist_mt[len(src_l):,:len(src_l)],axis=1)

            score = emd(_safe_divide(c1, np.sum(c1)),
                        _safe_divide(c2, np.sum(c2)),
                        dist_mt)

            #flow = np.array(flow)
            #inter_sentence = np.max(flow[:len(src_grams), len(src_grams):], axis=0)

            token_relevance = []
            if n_gram ==1:
                for l in range(0, len(hyp_l)):
                    token_relevance.append((hyp_l[l], inter_sentence[l]))
            if n_gram == 2:
                for l in range(0, len(hyp_considered) -1):
                    token_relevance.append((hyp_considered[l], min(inter_sentence[l-1],inter_sentence[l])))

            preds.append(1 - score)
            token_relevances.append(token_relevance)

    return preds, token_relevances