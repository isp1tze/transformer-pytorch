def gelu(x):
    out = 1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    return out * x / 2


class LayerNorm(nn.Module):
    """construct layernorm module"""

    def __init__(self, n_state, epsilon=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(n_state))
        self.beta = nn.Parameter(torch.zeros(n_state))
        self.epsilon = epsilon

    def forward(self, x):
        mean_ = x.mean((-1), keepdim=True)
        std = (x - mean_).pow(2).mean((-1), keepdim=True)
        x = (x - mean_) / torch.sqrt(std + self.epsilon)
        return self.gamma * x + self.beta


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        # print('***', x.size())
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size()[-1]), self.weight)
        return x.view(*size_out)


class Attention(nn.Module):
    """single block of attention"""

    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx

        assert n_state % config.num_head == 0

        '''
        self.register_buffer() is used to register a buffer that is not used
        as a model parameter
        '''
        self.register_buffer('masking', torch.tril(torch.ones(n_ctx, n_ctx)). \
                             view(1, 1, n_ctx, n_ctx))
        self.num_head = config.num_head
        self.split_size = n_state
        self.scale = scale

        # three different conv pre-attention layers
        self.conv_attn = Conv1D(nx, n_state)
        self.conv_proj = Conv1D(n_state, nx)

        # self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w /= np.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.masking[:, :, ns - nd:ns, :ns]
        w = w * b + 1e-9 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.num_head, x.size(-1) // self.num_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        '''x = self.conv_attn(x)
        try:
            query_, k_, v_ = x.split(self.split_size, dim=2)
        except ValueError as e:
            print('****', x.size())'''
        q1 = self.split_heads(self.conv_attn(query))
        k1 = self.split_heads(self.conv_attn(key), k=True)
        v1 = self.split_heads(self.conv_attn(value))

        # perform attention
        a = self._attn(q1, k1, v1)
        a = self.merge_heads(a)
        a = self.conv_proj(a)
        return a


class MLP(nn.Module):
    """MLP Module"""

    def __init__(self, hid_dim, config):
        super(MLP, self).__init__()
        nx = config.embedding_dim
        self.conv_fc = Conv1D(nx, hid_dim)
        self.conv_proj = Conv1D(hid_dim, nx)
        self.act = gelu

    def forward(self, x):
        out = self.act(self.conv_fc(x))
        out = self.conv_proj(out)
        return out


class Block(nn.Module):
    """One complete block / stack"""

    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.embedding_dim
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, config)
        self.ln2 = LayerNorm(nx)

    def forward(self, x):
        out = self.attn(x, x, x)
        out = self.ln1(out + x)
        mlp_out = self.mlp(out)
        out = self.ln2(out + mlp_out)
        return out


class TransformerModel(nn.Module):
    """Transformer Model"""

    def __init__(self, config, n_ctx=128):
        super(TransformerModel, self).__init__()
        self.n_ctx = n_ctx
        block = Block(n_ctx, config, scale=True)
        self.out = nn.ModuleList([copy.deepcopy(block) for _ in range(config.num_layers)])
        self.single_dim_proj = Conv1D(config.embedding_dim, 1)
        self.embed_proj = Conv1D(1, config.embedding_dim)

    def forward(self, x):
        # print(x.size)
        try:
            out = self.embed_proj(x)
        except RuntimeError:
            print('*** out.shape', x.size())
        # print(out.shape)
        # pass from the blocks
        for block in self.out:
            out = block(out)
            # print('**', out.shape)
        try:
            out = self.single_dim_proj(out).view(-1, self.n_ctx)
        except RuntimeError:
            print(out.size())
        return gelu(out)
