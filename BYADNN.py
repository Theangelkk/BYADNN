import torch
import torch.nn as nn
from torch.nn import functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam

import math

def custom_cat(dim1=0, dim2=0, dim3=0, list_tensors=[], device="cpu"):

  final_tensor = torch.zeros((dim1, dim2, dim3)).to(device)

  last_idx = 0
  for t in list_tensors:
    final_tensor[:, :, last_idx : last_idx + t.shape[2]] = t
    last_idx = t.shape[2]

  return final_tensor

# One Hot Encoding class
class ToOneHot(PyroModule):
    def __init__(self, num_classes, device="cpu"):
        super(ToOneHot, self).__init__()
        self.num_classes = num_classes
        self.device = device
    def forward(self, x):
        return F.one_hot(x.long(), num_classes=self.num_classes).type(torch.FloatTensor).to(self.device)

# PositionalEmbedding: It could be important in this case in order to maintain a sort of relashionship
# between elments of sequence
class PositionalEncoding(PyroModule):

  def __init__(self, d_model, max_len=5000):
    """
    Inputs
      d_model - Hidden dimensionality of the input.
      max_len - Maximum length of a sequence to expect.
    """
    super().__init__()

    # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
    # Used for tensors that need to be on the same device as the module.
    # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
    self.register_buffer('pe', pe, persistent=False)

  def forward(self, x):
    return x + self.pe[:, :x.size(1)]

# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
  assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
  if mask.ndim == 3:
      mask = mask.unsqueeze(1)
  while mask.ndim < 4:
      mask = mask.unsqueeze(0)
  return mask

def scaled_dot_product(q, k, v, mask=None):
  d_k = q.size()[-1]
  attn_logits = torch.matmul(q, k.transpose(-2, -1))
  attn_logits = attn_logits / math.sqrt(d_k)
        
  if mask is not None:
    attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        
  attention = F.softmax(attn_logits, dim=-1)
  values = torch.matmul(attention, v)
  return values, attention

class MultiheadAttention(PyroModule):

  def __init__(self, input_dim, embed_dim, num_heads):
    super().__init__()
    assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    # Stack all weight matrices 1...h together for efficiency
    # Note that in many implementations you see "bias=False" which is optional
    self.qkv_proj = PyroModule[nn.Linear](input_dim, 3*embed_dim)
    self.o_proj = PyroModule[nn.Linear](embed_dim, embed_dim)

    self._reset_parameters()

  def _reset_parameters(self):
        
    # Original Transformer initialization, see PyTorch documentation
    nn.init.xavier_uniform_(self.qkv_proj.weight)
    self.qkv_proj.bias.data.fill_(0)
    
    nn.init.xavier_uniform_(self.o_proj.weight)
    self.o_proj.bias.data.fill_(0)

  def forward(self, x, mask=None, return_attention=False):
    
    batch_size, seq_length, dim_input = x.size()
    
    if mask is not None:
      mask = expand_mask(mask)
    
    qkv = self.qkv_proj(x)

    # Separate Q, K, V from linear output
    qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
    qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
    q, k, v = qkv.chunk(3, dim=-1)

    # Determine value outputs
    values, attention = scaled_dot_product(q, k, v, mask=mask)
    values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
    values = values.reshape(batch_size, seq_length, self.embed_dim)
    o = self.o_proj(values)

    if return_attention:
      return o, attention
    else:
      return o

class EncoderBlock(PyroModule):

  def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
    """
    Inputs:
        input_dim - Dimensionality of the input
        num_heads - Number of heads to use in the attention block
        dim_feedforward - Dimensionality of the hidden layer in the MLP
        dropout - Dropout probability to use in the dropout layers
    """
    super().__init__()

    # Attention layer
    self.self_attn = PyroModule[MultiheadAttention](input_dim, input_dim, num_heads)

    # Two-layer MLP
    self.linear_net = PyroModule[nn.Sequential](
                                      PyroModule[nn.Linear](input_dim, dim_feedforward),
                                      PyroModule[nn.Dropout](dropout),
                                      PyroModule[nn.ReLU](inplace=True),
                                      PyroModule[nn.Linear](dim_feedforward, input_dim)
                                )

    # Layers to apply in between the main layers
    self.norm1 = PyroModule[nn.LayerNorm](input_dim)
    self.norm2 = PyroModule[nn.LayerNorm](input_dim)
    self.dropout = PyroModule[nn.Dropout](dropout)

  def forward(self, x, mask=None):
  
    # Attention part
    attn_out = self.self_attn(x, mask=mask)
    x = x + self.dropout(attn_out)
    x = self.norm1(x)

    # MLP part
    linear_out = self.linear_net(x)
    x = x + self.dropout(linear_out)
    x = self.norm2(x)

    return x

# Link: https://docs.pyro.ai/en/stable/nn.html
class Model(PyroModule):
  def __init__(self, len_dataset=100, input_features=1, seq_len=1, embedding_dim=30, num_enc_block=1, num_heads=3, h_enc_layer=50, dropout=0.0,
                h1=100, h2=50, add_positional_encoding=True, add_temporal_embedding=True, sigma_layer=True,
                device="cpu"):
    super().__init__()

    self.num_enc_block = num_enc_block
    self.num_heads = num_heads
    self.h_enc_layer = h_enc_layer
    self.dropout = dropout
    self.embedding_dim = embedding_dim
    self.h1 = h1
    self.h2 = h2
    self.device = device
    self.len_dataset = len_dataset
    self.input_features = input_features
    self.seq_len = seq_len
    self.add_positional_encoding = add_positional_encoding
    self.add_temporal_embedding = add_temporal_embedding
    self.sigma_layer = sigma_layer

    # Input dim -> Model dim ==> Embedding layer
    self.input_net = PyroModule[nn.Sequential](
                                                PyroModule[nn.Dropout](self.dropout),
                                                PyroModule[nn.Linear](self.input_features, self.embedding_dim)
    )
    
    if self.add_positional_encoding:
      self.positional_encoding = PyroModule[PositionalEncoding](d_model=self.embedding_dim, max_len=self.seq_len)

    # One-Hot Encoding Temporal layers
    if self.add_temporal_embedding:
      
      self.layer_one_hot_enc_hours = PyroModule[ToOneHot](num_classes=24, device=self.device)

      self.layer_one_hot_enc_days = PyroModule[ToOneHot](num_classes=31, device=self.device)

      self.layer_one_hot_enc_week_of_month = PyroModule[ToOneHot](num_classes=6, device=self.device)
                                                                      
      self.layer_one_hot_enc_months = PyroModule[ToOneHot](num_classes=12, device=self.device)
                                                                  
      # From 2014 - 2024
      self.layer_one_hot_enc_years = PyroModule[ToOneHot](num_classes=10, device=self.device)

      self.dim_temp = 24 + 31 + 6 + 12 + 10
      self.proj_temporal_emb = PyroModule[nn.Linear](self.dim_temp, self.embedding_dim)

      nn.init.xavier_uniform_(self.proj_temporal_emb.weight)
      self.proj_temporal_emb.bias.data.fill_(0)

    # Encoder blocks
    self.enc_blocks = PyroModule[nn.ModuleList]([PyroModule[EncoderBlock](self.embedding_dim, self.num_heads, self.h_enc_layer, self.dropout) for _ in range(self.num_enc_block)])

    self.fc1 = PyroModule[nn.Linear](self.embedding_dim * seq_len, h1)
    nn.init.xavier_uniform_(self.fc1.weight)
    self.fc1.bias.data.fill_(0)

    # Definizione dei pesi stocastici del layer Fully connected
    self.fc1.weight = PyroSample(dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).expand([h1, self.embedding_dim * seq_len]).to_event(2))
    self.fc1.bias = PyroSample(dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).expand([h1]).to_event(1))

    # Secondo layer Fully connected
    self.fc2 = PyroModule[nn.Linear](h1, h2)
    nn.init.xavier_uniform_(self.fc2.weight)
    self.fc2.bias.data.fill_(0)

    # Definizione dei pesi stocastici del layer Fully connected
    self.fc2.weight = PyroSample(dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).expand([h2, h1]).to_event(2))
    self.fc2.bias = PyroSample(dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).expand([h2]).to_event(1))

    # Terzo layer Fully connected
    self.fc3 = PyroModule[nn.Linear](h2, 1)
    nn.init.xavier_uniform_(self.fc3.weight)
    self.fc3.bias.data.fill_(0)

    # Definizione dei pesi stocastici del layer Fully connected
    self.fc3.weight = PyroSample(dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).expand([1, h2]).to_event(2))
    self.fc3.bias = PyroSample(dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).expand([1]).to_event(1))

    if self.sigma_layer:
      self.fc3_sigma = PyroModule[nn.Linear](h2, 1)
      nn.init.xavier_uniform_(self.fc3_sigma.weight)
      self.fc3_sigma.bias.data.fill_(0)

      # Definizione dei pesi stocastici del layer Fully connected
      self.fc3_sigma.weight = PyroSample(dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).expand([1, h2]).to_event(2))
      self.fc3_sigma.bias = PyroSample(dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).expand([1]).to_event(1))

    self.relu = PyroModule[nn.ReLU]()
    self.sigmoid = PyroModule[nn.Sigmoid]()

    self.encoder = PyroModule[nn.Sequential](
                                              self.fc1,
                                              self.relu,
                                              self.fc2,
                                              self.relu,

    )

    self.encoder_mu = PyroModule[nn.Sequential](
                                                self.fc3,
                                                self.sigmoid
    )

    if self.sigma_layer:
      self.encoder_sigma = PyroModule[nn.Sequential](
                                                      self.fc3_sigma,
                                                      self.sigmoid
      )

  def forward(self, x, date=None, mask=None, y=None):

    n_samples = x.shape[0]

    # Vuole: Batch_size, SeqLen, dim_input
    x = x.reshape((x.shape[0], self.seq_len, self.input_features))

    # Embedding layer 
    x = self.input_net(x)

    if self.add_positional_encoding:
      x = self.positional_encoding(x)

    if self.add_temporal_embedding:
  
      emb_hour = self.layer_one_hot_enc_hours(date[0])
      emb_day = self.layer_one_hot_enc_days(date[1])
      emb_week_of_month = self.layer_one_hot_enc_week_of_month(date[2])
      emb_month = self.layer_one_hot_enc_months(date[3])
      emb_year = self.layer_one_hot_enc_years(date[4])
      
      emb_temporal = torch.cat((emb_hour, emb_day, emb_week_of_month, emb_month, emb_year), 2).to(self.device)
      emb_temporal = self.proj_temporal_emb(emb_temporal)
      x = x + emb_temporal

    # Encoder blocks
    for l in self.enc_blocks:
      x = l(x, mask=mask)

    x = x.view(-1, self.seq_len * self.embedding_dim)

    # Estrazione della media (predizione) e sigma (varianza)
    enc = self.encoder(x)
    mu = self.encoder_mu(enc).squeeze()
    sigma = self.encoder_sigma(enc).squeeze()

    if self.sigma_layer:
      sigma = self.encoder_sigma(enc).squeeze()

      with pyro.plate("data", n_samples, device=self.device):
        sigma = torch.clamp(sigma, min=0.00000001, max=1.0)
        obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
    else:
      sigma = pyro.sample("sigma", dist.LogNormal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)))

      with pyro.plate("data", n_samples, device=self.device):
        obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    return mu