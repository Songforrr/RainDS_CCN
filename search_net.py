import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.optim as optim
import torchvision.transforms as transforms
import itertools
from PIL import Image
from copy import deepcopy
from torchvision import models
from torch.autograd import Variable
from se_nets import SEBasicBlock, InsNorm
from ops_derain import *
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')

class RB_s(nn.Module):     
    def __init__(self, in_dim=64, out_dim=64, res_dim=64, k1_size=3, k2_size=1, dilation=1, norm_type='batch_norm', with_relu=True):
        super(RB_s, self).__init__()
        
        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm1 = FeatNorm(norm_type, in_dim)        
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm2 = FeatNorm(norm_type, in_dim)

        self.up_conv = ConvLayer(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)
        self.up_norm = FeatNorm(norm_type, res_dim)

        self.se = SEBasicBlock(res_dim, res_dim, reduction=int(res_dim/2), with_norm=True)
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=k2_size, stride=1)
        self.down_norm = FeatNorm(norm_type, out_dim)
        
        self.with_relu = with_relu            
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x
        
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        x+= x_r
        x = self.relu(self.norm2(x))

        x = self.up_norm(self.up_conv(x))
        x+= res
        x = self.relu(x)
        res = x

        x = self.se(x)
        x = self.down_norm(self.down_conv(x))
        x+= x_r

        if self.with_relu:
            x = self.relu(x)
        else:
            pass
            
        return x, res


class SearchCell(nn.Module):
  """ Cell for search
  Each edge is mixed and continuous relaxed.
  """
  def __init__(self, C_pp, C_p, C, space_name, reduction_p, reduction, expansion_p, expansion, n_nodes=4):
    """
    Args:
      n_nodes: # of intermediate n_nodes
      C_pp: C_out[k-2]
      C_p : C_out[k-1]
      C   : C_in[k] (current)
      reduction_p: flag for whether the previous cell is reduction cell or not
      reduction: flag for whether the current cell is reduction cell or not
    """
    super().__init__()
    self.reduction = reduction
    self.expansion = expansion
    self.n_nodes = n_nodes

    # If previous cell is reduction cell, current input size does not match with
    # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
    if reduction_p   : self.preproc0 = FactorizedReduce(C_pp, C, 2, False)
    elif expansion_p : self.preproc0 = FactorizedExpand(C_pp, C, 1, False)
    else             : self.preproc0 = FactorizedReduce(C_pp, C, 1, False)
    self.preproc1 = StdConv(C_p, C, 1, 1, 0, affine=False)
    self.multiplier = self.n_nodes
    # generate dag
    self.dag = nn.ModuleList()
    for i in range(self.n_nodes):
      self.dag.append(nn.ModuleList())
      for j in range(2+i): # include 2 input nodes
        # reduction should be used only for input node
        stride = 2 if reduction and j < 2 else 1
        expand = True if expansion and j < 2 else False
        op     = MixedOp(C, stride, expand, space_name)
        self.PRIMITIVES  = op.PRIMITIVES
        self.dag[i].append(op)


  def forward(self, s0, s1, w_dag):
    s0 = self.preproc0(s0)
    s1 = self.preproc1(s1)
    states = [s0, s1]
    offset = 0
    for i in range(self.n_nodes):
      clist = []
      for j, h in enumerate(states):
        x = self.dag[i][j](h, w_dag[offset+j])
        clist.append( x )
      s = sum(clist)
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self.n_nodes:], dim=1)


class Raincleaner_search(nn.Module):
    def __init__(self, space_name, test_with_multigpus=False):
        super(Raincleaner_search, self).__init__()
        
        # Initial convolutional layers
        self.conv1 = ConvLayer(3, 64, kernel_size=3, stride=1)
        self.norm1 = FeatNorm("batch_norm", 64)
        self.conv2 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.norm2 = FeatNorm("batch_norm", 64)
        self.mult_gpu_test = test_with_multigpus

        cells = []
        self.block_number = 6
        self.n_nodes      = 4
        layer_reductions = [False] + [False] + [False]
        layer_expansions = [False] + [False] + [False]
        C_pp, C_p, C_curr = 64, 64, 64
        reduction_p, expansion_p = False, False
        for index, (reduction, expansion) in enumerate( zip(layer_reductions, layer_expansions)):
            cell = SearchCell(C_pp, C_p, C_curr, space_name, reduction_p, reduction, expansion_p, expansion, self.n_nodes) 
            reduction_p = reduction
            expansion_p = expansion
            cells.append(cell)
            C_pp, C_p = C_p, cell.multiplier*C_curr

        self.CPRIMITIVES = cell.PRIMITIVES
        self.cells = nn.ModuleList(cells)
        self._init_alphas()

        self.conv3 = ConvLayer(256, 64, kernel_size=3, stride=1)
        self.norm3 = FeatNorm("batch_norm", 64)        
        self.conv4 = ConvLayer(64, 3, kernel_size=3, stride=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def _init_alphas(self):
        k       = sum(1 for i in range(self.n_nodes) for n in range(2+i))
        n_ops   = len(self.CPRIMITIVES)
        self.alphas_normal = Parameter(torch.Tensor(k, n_ops))
        nn.init.normal_(self.alphas_normal, 0, 0.001)

    def get_alphas(self):
        return [self.alphas_normal]

    def get_weights(self):
        lists = list(self.conv1.parameters()) + list(self.norm1.parameters()) + list(self.conv2.parameters()) + list(self.norm2.parameters()) + list(self.cells.parameters()) + list(self.conv3.parameters()) + list(self.norm3.parameters()) + list(self.conv4.parameters())
        return lists

    def print_alphas(self):
        print('CPRIMITIVES : {:}'.format(self.CPRIMITIVES))
        print("Alphas_normal: {:}".format(torch.squeeze(self.alphas_normal)))

    def forward(self, x):        
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))  

        s0 = s1 = out
        for i, cell in enumerate(self.cells):
            weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.relu(self.norm3(self.conv3(s1)))                
        out = self.tanh(self.conv4(out))
        if self.mult_gpu_test:
            out = out.cuda(0)
            
        out = out + x

        return out        

    def genotype(self):

        def _parse(weights, PRIMITIVES):
            gene, n, start = [], 2, 0
            for i in range(self.n_nodes):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j, float(W[j][k_best])))
                start = end
                n += 1
            return gene


        with torch.no_grad():
            gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).cpu().numpy(), self.CPRIMITIVES)

            concat = range(2, self.n_nodes+2)
            genotype = Genotype(
                normal=gene_normal, normal_concat=concat,
            )
            return genotype



        
#---------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)        
        return out
        
        
class FeatNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super(FeatNorm, self).__init__()
        if norm_type == "instance":
            self.norm = InsNorm(dim)
        elif norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(dim)
        else:
            raise Exception("Normalization type incorrect.")

    def forward(self, x):
        out = self.norm(x)        
        return out

