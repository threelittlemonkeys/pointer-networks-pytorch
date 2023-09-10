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
