from torch import nn

from modelling.functional import TransformerDecoderLayer, BaseTransformerLayer
from modelling.positional_encoding import PositionalEncoding



class Transformer(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model, 
                 n_heads, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dim_feedforward, 
                 dropout, 
                 max_len):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len

        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        self.dropout = nn.Dropout(dropout)

        self.transformer_encoder = nn.ModuleDict({
            f"encoder_layer_{i}": BaseTransformerLayer(d_model, n_heads, dim_feedforward, dropout) 
            for i in range(num_encoder_layers)
        })

        self.transformer_decoder = nn.ModuleDict({
            f"decoder_layer_{i}": TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout) 
            for i in range(num_decoder_layers)
        })

        self.head = nn.Linear(d_model, vocab_size, bias=False)
    

        self.src_embedding.weight = self.tgt_embedding.weight
        self.head.weight = self.tgt_embedding.weight


    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.src_embedding(src)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        for encoder_layer in self.transformer_encoder.values():
            src = encoder_layer(src, src_mask)

        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)
        for decoder_layer in self.transformer_decoder.values():
            tgt = decoder_layer(tgt, src, tgt_mask, src_mask)

        return self.head(tgt)