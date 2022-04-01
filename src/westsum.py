from sqlite3 import DataError
from transformers import DistilBertTokenizer, AlbertTokenizer
import torch
from models.model_builder import ExtSummarizer, HiWestSummarizer
from pathlib import Path
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def summarize(input_data, num_sents=3, model='hiwest', device='cpu'):
    if model == 'hiwestdistil':     
        ## TODO: Add other tokenizer and models
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
        model = load_model('hiwestdistilbert')
    elif model == 'bertsum':
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
        model = load_model('bertsum')
    else:
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)
        model = load_model('hiwestalbert')
    # else:
    #     tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)

    
    input_data, sents = preprocess(input_data)
    source_data = process_text(input_data, tokenizer, device)
    selected_ids, sent_scores = get_scores_summary(model, source_data, num_sents, device=device)
    sentences = []

    try:
        assert(len(sents) == len(sent_scores[0]))
    except:
        raise DataError('Input format incorrect. Please check for proper sentence structures.')
    for i, text in enumerate(sents):
        sent = {
            "text": text,
            "scores": sent_scores[0][i]
        }
        sentences.append(sent)

    return sentences, np.sort(selected_ids)


def load_model(model='bertsum', device='cpu'):
    print(f'Loading model-{model}... ')
    
    print("Checkpoint loaded.")
    if model == 'bertsum':
        checkpoint = torch.load(BASE_DIR / 'checkpoints/bertsum.pt', map_location='cpu')["model"]
        model = ExtSummarizer(device=device, checkpoint=checkpoint, bert_type='distilbert').to(device)
    elif model == 'hiwestalbert':
        checkpoint = torch.load(BASE_DIR / 'checkpoints/hiwest.pt', map_location='cpu')["model"]
        model = HiWestSummarizer(device=device, checkpoint=checkpoint, bert_type='albert').to(device)
    else:
        checkpoint = torch.load(BASE_DIR / 'checkpoints/hiwest_distil.pt', map_location='cpu')["model"]
        model = HiWestSummarizer(device=device, checkpoint=checkpoint, bert_type='distilbert').to(device)
    return model


def preprocess(input_data):
    """
    - Remove \n
    - Sentence Tokenize
    - Add [SEP] [CLS] as sentence boundary
    """
    raw_text = input_data.replace("\n", " ").replace("[CLS] [SEP]", " ")
    sents = sent_tokenize(raw_text)
    processed_text = "[CLS] [SEP]".join(sents)
    return processed_text, sents

def process_text(processed_text, tokenizer, device='cpu', max_pos=512):
    print(f'Processing text... ')
    if tokenizer.name_or_path == 'albert-base-v2':
        tokenizer.vocab = tokenizer.get_vocab()
        sep_vid = tokenizer.vocab["[SEP]"]
        cls_vid = tokenizer.vocab["[CLS]"]
    else:
        sep_vid = tokenizer.vocab["[SEP]"]
        cls_vid = tokenizer.vocab["[CLS]"]

    print(sep_vid)
    print(cls_vid)


    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace("[cls]", "[CLS]").replace("[sep]", "[SEP]")
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        
        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0
        return src, mask_src, segments_ids, clss, mask_cls

    src, mask_src, segments_ids, clss, mask_cls = _process_src(processed_text)
    segs = torch.tensor(segments_ids)[None, :].to(device)
    src_text = [[sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")]]
    return src, mask_src, segs, clss, mask_cls, src_text


def get_scores_summary(model, input_data, max_length=3, device='cpu'):
    print("Generating scores...")
    with torch.no_grad():
        src, mask, segs, clss, mask_cls, src_str = input_data
        sent_scores, mask = model(src, segs, clss, mask, mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)

        print(selected_ids)

        return selected_ids[0][:max_length], sent_scores

def try_ftp():
    print("abc")
