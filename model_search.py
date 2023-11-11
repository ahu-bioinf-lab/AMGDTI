import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Op(nn.Module):
    def __init__(self):
        super(Op, self).__init__()

    def forward(self, x, adjs, ws, idx):
        return ws[idx]*torch.spmm(adjs[idx], x)

class Cell(nn.Module):

    def __init__(self, n_step, n_hid_prev, n_hid, cstr, use_norm = True, use_nl = True):
        super(Cell, self).__init__()
        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step
        self.norm = nn.LayerNorm(n_hid, elementwise_affine = False) if use_norm is True else lambda x : x
        self.use_nl = use_nl
        assert(isinstance(cstr, list))
        self.cstr = cstr

        self.ops_seq = nn.ModuleList()
        for i in range(1, self.n_step):
            self.ops_seq.append(Op())

        self.ops_res = nn.ModuleList()
        for i in range(2, self.n_step):
            for j in range(i - 1):
                self.ops_res.append(Op())

        self.last_seq = Op()
        self.last_res = nn.ModuleList()
        for i in range(self.n_step - 1):
            self.last_res.append(Op())
    

    def forward(self, x, adjs, ws_seq, idxes_seq, ws_res, idxes_res):
        x = self.affine(x)
        states = [x]
        offset = 0
        for i in range(self.n_step - 1):
            seqi = self.ops_seq[i](states[i], adjs[:-1], ws_seq[0][i], idxes_seq[0][i])   #! 排除零运算
            resi = sum(self.ops_res[offset + j](h, adjs, ws_res[0][offset + j], idxes_res[0][offset + j]) for j, h in enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)

        adjs_cstr = [adjs[i] for i in self.cstr]
        out_seq = self.last_seq(states[-1], adjs_cstr, ws_seq[1], idxes_seq[1])
        adjs_cstr.append(adjs[-1])
        out_res = sum(self.last_res[i](h, adjs_cstr, ws_res[1][i], idxes_res[1][i]) for i, h in enumerate(states[:-1]))
        output = self.norm(out_seq + out_res)
        if self.use_nl:
            output = F.gelu(output)
        return output


class Model(nn.Module):
    def __init__(self, in_dims, n_hid, n_adjs, n_steps, cstr, attn_dim = 64, use_norm = True, out_nl = True):
        super(Model, self).__init__()
        self.cstr = cstr
        self.n_adjs = n_adjs
        self.n_hid = n_hid
        self.ws = nn.ModuleList()

        assert(isinstance(in_dims, list))
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(64, n_hid))
        assert(isinstance(n_steps, list))

        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, cstr, use_norm = use_norm, use_nl = out_nl))

        self.as_seq = []
        self.as_last_seq = []
        for i in range(len(n_steps)):
            if n_steps[i] > 1:
                ai = 1e-3 * torch.randn(n_steps[i] - 1, n_adjs - 1)
                ai = ai.cuda()
                ai.requires_grad_(True)
                self.as_seq.append(ai)
            else:
                self.as_seq.append(None)
            ai_last = 1e-3 * torch.randn(len(cstr))
            ai_last = ai_last.cuda()
            ai_last.requires_grad_(True)
            self.as_last_seq.append(ai_last)
        
        ks = [sum(1 for i in range(2, n_steps[k]) for j in range(i - 1)) for k in range(len(n_steps))]
        self.as_res = []
        self.as_last_res = []

        for i in range(len(n_steps)):
            if ks[i] > 0:
                ai = 1e-3 * torch.randn(ks[i], n_adjs)
                ai = ai.cuda()
                ai.requires_grad_(True)
                self.as_res.append(ai)
            else:
                self.as_res.append(None)
            
            if n_steps[i] > 1:
                ai_last = 1e-3 * torch.randn(n_steps[i] - 1, len(cstr) + 1)
                ai_last = ai_last.cuda()
                ai_last.requires_grad_(True)
                self.as_last_res.append(ai_last)
            else:
                self.as_last_res.append(None)
        
        assert(ks[0] + n_steps[0] + (0 if self.as_last_res[0] is None else self.as_last_res[0].size(0)) == (1 + n_steps[0]) * n_steps[0] // 2)

        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)

    def alphas(self):
        alphas = []
        for each in self.as_seq:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_seq:
            alphas.append(each)
        for each in self.as_res:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_res:
            if each is not None:
                alphas.append(each)
        return alphas
    
    def sample(self, eps):

        idxes_seq = []
        idxes_res = []
        if np.random.uniform() < eps:
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.randint(low=0, high=self.as_seq[i].size(-1), size=self.as_seq[i].size()[:-1]).cuda())
                temp.append(torch.randint(low=0, high=self.as_last_seq[i].size(-1), size=(1,)).cuda())
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.randint(low=0, high=self.as_res[i].size(-1), size=self.as_res[i].size()[:-1]).cuda())
                temp.append(None if self.as_last_res[i] is None else torch.randint(low=0, high=self.as_last_res[i].size(-1), size=self.as_last_res[i].size()[:-1]).cuda())
                idxes_res.append(temp)
        else:
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.argmax(F.softmax(self.as_seq[i], dim=-1), dim=-1))
                temp.append(torch.argmax(F.softmax(self.as_last_seq[i], dim=-1), dim=-1))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.argmax(F.softmax(self.as_res[i], dim=-1), dim=-1))
                temp.append(None if self.as_last_res[i] is None else torch.argmax(F.softmax(self.as_last_res[i], dim=-1), dim=-1))
                idxes_res.append(temp)
        return idxes_seq, idxes_res
    
    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res):
        hid = torch.zeros((node_types.size(0), self.n_hid)).cuda()
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        temps = []; attns = []
        for i, meta in enumerate(self.metas):
            ws_seq = []
            ws_seq.append(None if self.as_seq[i] is None else F.softmax(self.as_seq[i], dim=-1))
            ws_seq.append(F.softmax(self.as_last_seq[i], dim=-1))
            ws_res = []
            ws_res.append(None if self.as_res[i] is None else F.softmax(self.as_res[i], dim=-1))
            ws_res.append(None if self.as_last_res[i] is None else F.softmax(self.as_last_res[i], dim=-1))
            hidi = meta(hid, adjs, ws_seq, idxes_seq[i], ws_res, idxes_res[i])
            temps.append(hidi)
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
            attns.append(attni)

        hids = torch.stack(temps, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        return out
    
    def parse(self):
        idxes_seq, idxes_res = self.sample(0.)
        msg_seq = []; msg_res = []
        for i in range(len(idxes_seq)):
            map_seq = [self.cstr[idxes_seq[i][1].item()]]
            msg_seq.append(map_seq if idxes_seq[i][0] is None else idxes_seq[i][0].tolist() + map_seq)
            assert(len(msg_seq[i]) == self.metas[i].n_step)

            temp_res = []
            if idxes_res[i][1] is not None:
                for item in idxes_res[i][1].tolist():
                    if item < len(self.cstr):
                        temp_res.append(self.cstr[item])
                    else:
                        assert(item == len(self.cstr))
                        temp_res.append(self.n_adjs - 1)
                if idxes_res[i][0] is not None:
                    temp_res = idxes_res[i][0].tolist() + temp_res
            assert(len(temp_res) == self.metas[i].n_step * (self.metas[i].n_step - 1) // 2)
            msg_res.append(temp_res)
        
        return msg_seq, msg_res