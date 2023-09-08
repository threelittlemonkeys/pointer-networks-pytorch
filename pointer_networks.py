from utils import *
from embedding import *
from rnn_encoder import *
from rnn_decoder import *

class pointer_networks(nn.Module):

    def __init__(self, cti, wti):

        super().__init__()

        # architecture
        self.enc = rnn_encoder(cti, wti)
        self.dec = rnn_decoder(cti, wti)
        if CUDA: self = self.cuda()

    def init_state(self, b):

        self.dec.h = zeros(b, 1, HIDDEN_SIZE)
        self.dec.attn.W = zeros(b, 1, self.dec.M.size(1))

    def forward(self, xc, xw, y0): # for training

        self.zero_grad()
        b = len(y0) # batch size
        loss = 0
        mask, lens = maskset(y0)

        self.dec.M, self.dec.H = self.enc(xc, xw, lens)
        self.init_state(b)
        yc = LongTensor([[[SOS_IDX]]] * b)
        yw = LongTensor([[SOS_IDX]] * b)

        for t in range(y0.size(1)):
            yo = self.dec(yc, yw, mask)
            y1 = y0[:, t] - 1 # teacher forcing
            loss += F.nll_loss(yo, y1, ignore_index = PAD_IDX - 1)
            yc = torch.cat([xc[i, j] for i, j in enumerate(y1)]).view(b, 1, -1)
            yw = LongTensor([xw[i, j] for i, j in enumerate(y1)]).unsqueeze(1)

        loss /= y0.size(1) # average over timesteps

        return loss

class rnn_encoder(nn.Module):

    def __init__(self, cti, wti):

        super().__init__()

        # architecture
        self.embed = embed(EMBED, cti, wti, batch_first = True, hre = HRE)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )

    def init_state(self, b): # initialize states

        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "GRU":
            return hs
        cs = zeros(n, b, h) # LSTM cell state
        return (hs, cs)

    def forward(self, xc, xw, lens):

        b = len(lens)
        s = self.init_state(b)
        lens = lens.cpu()

        x = self.embed(b, xc, xw)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True, enforce_sorted = False)
        h, s = self.rnn(x, s)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)

        return h, s

class rnn_decoder(nn.Module):

    def __init__(self, cti, wti):

        super().__init__()
        self.hs = None # source hidden state
        self.hidden = None # decoder hidden state

        self.M = None # encoder hidden states
        self.H = None # decoder hidden states
        self.h = None # decoder output

        # architecture
        self.embed = embed(EMBED, cti, wti, batch_first = True)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.attn = attn()

    def forward(self, yc, yw, mask):

        x = self.embed(EMBED, yc, yw)
        h, self.H = self.rnn(x, self.H)
        y = self.attn(self.M, h, mask)

        return y

class attn(nn.Module): # content based input attention

    def __init__(self):

        super().__init__()

        # architecture
        self.W1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.W2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.V = nn.Linear(HIDDEN_SIZE, 1)
        self.W = None # attention weights

    def forward(self, hs, ht, mask):

        u = self.V(torch.tanh(self.W1(hs) + self.W2(ht))) # [B, L, H] -> [B, L, 1]
        u = u.squeeze(2).masked_fill(mask, -10000)
        self.W = F.log_softmax(u, 1) # [B, L]

        return self.W
