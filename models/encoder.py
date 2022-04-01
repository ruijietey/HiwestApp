import math
import torch
import torch.nn as nn
from models.neural import MultiHeadedAttention, PositionwiseFeedForward


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super().__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_inter_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class ExtLayer(nn.Module):
    def __init__(self, transformer, bert_used, d_model, d_ff, heads, dropout, num_inter_layers=0, doc_weight=0.4, extra_attention=False):
        super(ExtLayer, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.extra_attention = extra_attention
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer = transformer
        self.bert_used = bert_used
        self.doc_weight = doc_weight
        if bert_used == 'distilbert':
            self.transformer_inter = transformer.layer
        elif bert_used == 'albert':
            self.transformer_inter = transformer.albert_layer_groups
        else:
            self.transformer_inter = nn.ModuleList(
                [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
                 for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask, head_mask=None):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        if self.bert_used == 'distilbert':
            head_mask = [None] * self.transformer.n_layers if head_mask is None else head_mask
            x, = self.transformer(x, mask, head_mask) # Doc level representation
            # for i in range(self.transformer.n_layers):
            #     x, = self.transformer_inter[i](x, mask)  # all_sents * max_tokens * dim (Modified Masking for Pytorch above v1.2)
        elif self.bert_used == 'albert':
            head_mask = [None] * self.transformer.config.num_hidden_layers if head_mask is None else head_mask
            for i in range(self.transformer.config.num_hidden_layers):
                layers_per_group = int(self.transformer.config.num_hidden_layers / self.transformer.config.num_hidden_groups)
                group_idx = int(i / (self.transformer.config.num_hidden_layers / self.transformer.config.num_hidden_groups))
                # Attention mask below follows albert model way of extension
                extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                layer_group_output = self.transformer_inter[group_idx](x, extended_attention_mask, head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group])
                x = layer_group_output[0]

        x = (1-self.doc_weight)*top_vecs + (self.doc_weight)*x
        if self.extra_attention == True:
            x = self.global_attention(x)
        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores