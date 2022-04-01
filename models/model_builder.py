import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import AlbertConfig, BertModel, BertConfig, DistilBertConfig, DistilBertModel, AlbertModel
from models.encoder import ExtTransformerEncoder, ExtLayer


class Bert(nn.Module):
    def __init__(self, bert_type='distilbert'):
        super(Bert, self).__init__()
        print(f'Initiating BERT - ', bert_type)
        self.bert_type = bert_type

        if bert_type == 'bertbase':
            configuration = BertConfig()
            self.model = BertModel(configuration)
        elif bert_type == 'distilbert':
            configuration = DistilBertConfig()
            self.model = DistilBertModel(configuration)     
        elif bert_type == 'albert':
            self.model = AlbertModel.from_pretrained('albert-base-v2')  


    def forward(self, x, segs, mask):
        if self.bert_type == 'distilbert':
            top_vec = self.model(input_ids=x, attention_mask=mask)[0]
        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)[0]
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, device, checkpoint=None, bert_type='distilbert'):
        super().__init__()
        print("Initiating Ext Summ")
        self.device = device
        self.bert = Bert(bert_type='distilbert')
        self.ext_layer = ExtTransformerEncoder(
            self.bert.model.config.hidden_size, d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2
        )

        if checkpoint is not None:
            self.load_state_dict(checkpoint, strict=True)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        # print(torch.arange(top_vec.size(0)).unsqueeze(1).size())
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1).long(), clss.long()]
        # print(sents_vec.size())
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls



class HiWestSummarizer(nn.Module):
    def __init__(self, bert_type, device, checkpoint=None):
        super(HiWestSummarizer, self).__init__()
        self.device = device
        
        # Modification: Use same transformer for weight-sharing purpose
        self.sharing = True
        print(bert_type)

        if bert_type == 'distilbert':
            self.bert = Bert(bert_type='distilbert') # Modified: Add `args.other_bert`
            self.transformer = self.bert.model.transformer
            self.ext_layer = ExtLayer(self.transformer, bert_type, self.bert.model.config.hidden_size,
                    d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2, doc_weight=0.5, extra_attention=False)
        else:
            self.bert = Bert(bert_type='albert')
            self.transformer = self.bert.model.encoder  # For BERT, ALBERT, etc.
            self.ext_layer = ExtLayer(self.transformer, bert_type, self.bert.model.config.hidden_size,
                    d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2, doc_weight=0.4, extra_attention=False)
            
        if checkpoint is not None:
            self.load_state_dict(checkpoint)
        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        if self.sharing == False:
            top_vec = self.bert(src, segs, mask_src)
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
            sents_vec = sents_vec * mask_cls[:, :, None].float()
            sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        else:
            # top_vec, cls_vec = self.bert(src, segs, mask_src)
            top_vec = self.bert(src, segs, mask_src)
            cls_vec = top_vec[:, 0, :].unsqueeze(1)
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
            sents_vec = sents_vec * mask_cls[:, :, None].float()
            sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
