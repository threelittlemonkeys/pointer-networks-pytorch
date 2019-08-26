from utils import *
from embedding import embed

class encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # architecture
        self.embed = embed(-1, vocab_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = sum(EMBED.values()),
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )

        if CUDA:
            self = self.cuda()

    def init_state(self): # initialize RNN states
        args = (NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS)
        hs = zeros(*args) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(*args) # LSTM cell state
            return (hs, cs)
        return hs


    def forward(self, x, mask):
        self.hidden = self.init_state()
        x = self.embed(None, x)
        x = nn.utils.rnn.pack_padded_sequence(x, mask[1], batch_first = True)
        h, _ = self.rnn(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        return h

class decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.feed_input = True # input feeding

        # architecture
        self.embed = embed(-1, vocab_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = sum(EMBED.values()),
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.attn = attn()
        self.out = nn.Linear(HIDDEN_SIZE, vocab_size)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def forward(self, dec_in, enc_out, t, mask):
        x = self.embed(None, dec_in)
        h, _ = self.rnn(x, self.hidden)
        if self.attn:
            h = self.attn(h, enc_out, t, mask)
        h = self.out(h).squeeze(1)
        y = self.softmax(h)
        return y

class attn(nn.Module): # content based input attention
    def __init__(self):
        super().__init__()
        self.a = None # attention weights (for heatmap)
        self.h = None # attentional vector (for input feeding)

        # architecture
        self.w1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.w2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.softmax = nn.Softmax(2)

    def align(self, ht, hs, mask):
        a = self.w1(hs) + self.w2(ht)
        exit()
        a = ht.bmm(self.w1(hs).transpose(1, 2)) # [B, L, H] @ # TODO
        a = a.masked_fill(mask.unsqueeze(1), -10000) # masking in log space
        a = self.softmax(a) # [B, 1, H] @ [B, H, L] = [B, 1, L]
        return a # alignment vector as attention weights

    def forward(self, ht, hs, t, mask):
        mask = mask[0]
        a = self.a = self.align(ht, hs, mask)
        c = a.bmm(hs) # context vector [B, 1, L] @ [B, L, H] = [B, 1, H]
        h = self.h = torch.tanh(self.Wc(torch.cat((c, ht), 2)))
        return h # attentional vector as attentional hiodden state
