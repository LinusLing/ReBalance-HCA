# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, soft_cross_entropy_loss, \
    Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias, SpuerFrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext, ScaledDotProductAttention, MultiHeadAttention
from .model_transformer_super import TransformerContext as TransformerContextSuper
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from scipy.special import softmax as spysoftmax
from maskrcnn_benchmark.layers.gcn.gcn_layers import GCN
from maskrcnn_benchmark.layers.gcn._utils import adj_normalize
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors,\
    to_onehot, nms_overlaps, encode_box_info
import json
import random
from .loss import make_roi_relation_loss_evaluator
'-----------------------------------------------------------------------------------'
class FrequencyBias_GCL(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """
    '''
    该函数可用来替换roi_relation_predictors.py中的self.freq_bias()方法

    我们增加了predicate_all_list，这是一个51维的向量，用来囊括所有目标的谓词，如果该谓词在这个类中，则
    设置其的值大于0.例如对于原6分类，共有6个值大于0.其中on是第31位，在6分类中是第5位，则设置
    predicate_all_list[31]=5，以此类推
    '''

    def __init__(self, cfg, statistics, Dataset_choice, eps=1e-3, predicate_all_list=None):
        super(FrequencyBias_GCL, self).__init__()
        assert predicate_all_list is not None
        if Dataset_choice == 'VG':
            self.num_obj_cls = 151
        elif Dataset_choice == 'GQA_200':
            self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
        # self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = max(predicate_all_list) + 1
        old_matrix = statistics['fg_matrix'].float()

        fg_matrix = torch.zeros([self.num_obj_cls, self.num_obj_cls, self.num_rel_cls],
                                dtype=old_matrix.dtype, device=old_matrix.device)

        lines = 0
        assert len(predicate_all_list) == 51 or len(predicate_all_list) == 101
        for i in range(len(predicate_all_list)):
            if i == 0 or predicate_all_list[i] > 0:
                fg_matrix[:, :, lines] = old_matrix[:, :, i]
                lines = lines + 1
        assert lines == self.num_rel_cls

        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        '''以下是原函数，以上是我改的部分'''
        # pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs * self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2]
        :return:
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:, :, 0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:, :,
                                                                                    1].contiguous().view(batch_size, 1,
                                                                                                         num_obj)

        return joint_prob.view(batch_size, num_obj * num_obj) @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, dim_v)
            attn (bsz, len_q, len_k)
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, d_model)
            attn (bsz, len_q, len_k)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size() # len_k==len_v

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Merge adjacent information. Equal to linear layer if kernel size is 1
        Args:
            x (bsz, len, dim)
        Returns:
            output (bsz, len, dim)
        """
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask.float()

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask.float()

        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, input_feats, num_objs):
        """
        Args:
            input_feats [Tensor] (#total_box, d_model) : bounding box features of a batch
            num_objs [list of int] (bsz, ) : number of bounding box of each image
        Returns:
            enc_output [Tensor] (#total_box, d_model)
        """
        original_input_feats = input_feats
        input_feats = input_feats.split(num_objs, dim=0)
        input_feats = nn.utils.rnn.pad_sequence(input_feats, batch_first=True)

        # -- Prepare masks
        bsz = len(num_objs)
        device = input_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(1).expand(-1, pad_len, -1) # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_objs_).unsqueeze(-1) # (bsz, pad_len, 1)

        # -- Forward
        enc_output = input_feats
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output


class TransformerContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE   
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER      
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER        
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD         
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM     
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM         
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM    


        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj = nn.Linear(self.in_channels + self.embed_dim + 128, self.hidden_dim)
        self.lin_edge = nn.Linear(self.embed_dim + self.hidden_dim + self.in_channels, self.hidden_dim)
        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)
        self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim, 
                                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)
        self.context_edge = TransformerEncoder(self.edge_layer, self.num_head, self.k_dim, 
                                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

    
    def forward(self, roi_features, proposals, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        
        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer
        obj_pre_rep = cat((roi_features, obj_embed, pos_embed), -1)
        num_objs = [len(p) for p in proposals]
        obj_pre_rep = self.lin_obj(obj_pre_rep)
        obj_feats = self.context_obj(obj_pre_rep, num_objs)

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_labels)), dim=-1)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_preds)), dim=-1)

        # edge context
        edge_pre_rep = self.lin_edge(edge_pre_rep)
        edge_ctx = self.context_edge(edge_pre_rep, num_objs)

        return obj_dists, obj_preds, edge_ctx

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


class Single_Att_Layer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Single_Att_Layer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q_input, k_input, v_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            q_input, k_input, v_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask.float()

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask.float()

        return enc_output, enc_slf_attn

class Self_Attention_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, input_feats, num_objs):

        input_feats = input_feats.split(num_objs, dim=0)
        input_feats = nn.utils.rnn.pad_sequence(input_feats, batch_first=True)

        # -- Prepare masks
        bsz = len(num_objs)
        device = input_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(1).expand(-1, pad_len, -1) # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_objs_).unsqueeze(-1) # (bsz, pad_len, 1)

        # -- Forward
        enc_output, enc_slf_attn = self.transformer_layer(
                input_feats, input_feats, input_feats,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output

class Cross_Attention_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, visual_feats, textual_feats, num_objs):

        visual_feats = visual_feats.split(num_objs, dim=0)
        visual_feats = nn.utils.rnn.pad_sequence(visual_feats, batch_first=True)
        textual_feats = textual_feats.split(num_objs, dim=0)
        textual_feats = nn.utils.rnn.pad_sequence(textual_feats, batch_first=True)

        # -- Prepare masks
        bsz = len(num_objs)
        device = visual_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(1).expand(-1, pad_len, -1) # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_objs_).unsqueeze(-1) # (bsz, pad_len, 1)

        # -- Forward
        enc_output, enc_slf_attn = self.transformer_layer(
                visual_feats, textual_feats, textual_feats,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output


class Self_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Self_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.SA_transformer_encoder = Self_Attention_Encoder(self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats=None, num_objs=None):
        assert num_objs is not None
        outp = self.SA_transformer_encoder(x, num_objs)

        return outp

class Cross_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Cross_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.CA_transformer_encoder = Cross_Attention_Encoder(self.num_head, self.k_dim,
                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats, num_objs=None):
        assert num_objs is not None
        outp = self.CA_transformer_encoder(x, textual_feats, num_objs)

        return outp

# class Single_Layer_Hybrid_Attention(nn.Module):
#     """
#     A encoder model with self attention mechanism.
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.SA_Cell_vis = Self_Attention_Cell(config)
#         self.SA_Cell_txt = Self_Attention_Cell(config)
#         self.CA_Cell_vis = Cross_Attention_Cell(config)
#         self.CA_Cell_txt = Cross_Attention_Cell(config)

#     def forward(self, visual_feats, text_feats, num_objs):
#         tsa = self.SA_Cell_txt(text_feats, num_objs=num_objs)
#         tca = self.CA_Cell_txt(text_feats, visual_feats, num_objs=num_objs)
#         vsa = self.SA_Cell_vis(visual_feats, num_objs=num_objs)
#         vca = self.CA_Cell_vis(visual_feats, text_feats, num_objs=num_objs)
#         textual_output = tsa + tca
#         visual_output = vsa + vca

#         return visual_output, textual_output

class Single_Layer_Hybrid_Attention_Encoder_Decoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.SA_Cell_vis = Self_Attention_Cell(config)
        self.SA_Cell_txt = Self_Attention_Cell(config)
        self.CA_Cell_vis = Cross_Attention_Cell(config)
        self.CA_Cell_txt = Cross_Attention_Cell(config)

    def forward(self, tsa, tca, vsa, vca, num_objs):
        tsa = self.SA_Cell_txt(tsa, num_objs=num_objs)
        tca = self.CA_Cell_txt(tca, vsa, num_objs=num_objs)
        vsa = self.SA_Cell_vis(vsa, num_objs=num_objs)
        vca = self.CA_Cell_vis(vca, tsa, num_objs=num_objs)
        # tca = tsa + tca
        # vca = vsa + vca

        return tsa, tca, vsa, vca

class Single_Layer_Hybrid_Attention_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.SA_Cell_vis = Self_Attention_Cell(config)
        self.SA_Cell_txt = Self_Attention_Cell(config)
        self.CA_Cell_vis = Cross_Attention_Cell(config)
        self.CA_Cell_txt = Cross_Attention_Cell(config)

    def forward(self, visual_feats, text_feats, num_objs):
        tsa = self.SA_Cell_txt(text_feats, num_objs=num_objs)
        # tca = self.CA_Cell_txt(text_feats, visual_feats, num_objs=num_objs)
        vsa = self.SA_Cell_vis(visual_feats, num_objs=num_objs)
        # vca = self.CA_Cell_vis(visual_feats, text_feats, num_objs=num_objs)
        textual_output = tsa
        visual_output = vsa

        return visual_output, textual_output

class Single_Layer_Hybrid_Attention_Decoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.SA_Cell_vis = Self_Attention_Cell(config)
        self.SA_Cell_txt = Self_Attention_Cell(config)
        self.CA_Cell_vis = Cross_Attention_Cell(config)
        self.CA_Cell_txt = Cross_Attention_Cell(config)

    def forward(self, visual_feats, text_feats, num_objs):
        tsa = self.SA_Cell_txt(text_feats, num_objs=num_objs)
        tca = self.CA_Cell_txt(text_feats, visual_feats, num_objs=num_objs)
        vsa = self.SA_Cell_vis(visual_feats, num_objs=num_objs)
        vca = self.CA_Cell_vis(visual_feats, text_feats, num_objs=num_objs)
        textual_output = tsa + tca
        visual_output = vsa + vca

        return visual_output, textual_output


class SHA_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config, n_layers):
        super().__init__()
        self.cfg = config
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        self.cross_module = nn.ModuleList([
            # Single_Layer_Hybrid_Attention_Encoder(config)
            Single_Layer_Hybrid_Attention_Encoder_Decoder(config)
            for _ in range(n_layers)])

    def forward(self, visual_feats, text_feats, num_objs):
        visual_output = visual_feats
        textual_output = text_feats

        tsa = text_feats
        tca = text_feats
        vsa = visual_feats
        vca = visual_feats

        for enc_layer in self.cross_module:
            tsa, tca, vsa, vca = enc_layer(tsa, tca, vsa, vca, num_objs)
        tca = tca + tsa
        vca = vca + vsa
        vca = vca + tca

        return vca, tca

# class SHA_Encoder(nn.Module):
#     """
#     A encoder model with self attention mechanism.
#     """
#     def __init__(self, config, n_layers):
#         super().__init__()
#         self.cfg = config
#         self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
#         self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
#         self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
#         self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
#         self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
#         self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
#         self.cross_module = nn.ModuleList([
#             Single_Layer_Hybrid_Attention(config)
#             for _ in range(n_layers)])

#     def forward(self, visual_feats, text_feats, num_objs):
#         visual_output = visual_feats
#         textual_output = text_feats

#         for enc_layer in self.cross_module:
#             visual_output, textual_output = enc_layer(visual_output, textual_output, num_objs)

#         visual_output = visual_output + textual_output

#         return visual_output, textual_output

class SHA_Decoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config, n_layers):
        super().__init__()
        self.cfg = config
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        self.cross_module = nn.ModuleList([
            Single_Layer_Hybrid_Attention_Decoder(config)
            for _ in range(n_layers)])

    def forward(self, visual_feats, text_feats, num_objs):
        visual_output = visual_feats
        textual_output = text_feats

        for enc_layer in self.cross_module:
            visual_output, textual_output = enc_layer(visual_output, textual_output, num_objs)

        visual_output = visual_output + textual_output

        return visual_output, textual_output

class SHA_Context(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj_visual = nn.Linear(self.in_channels + 128, self.hidden_dim)
        self.lin_obj_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.lin_edge_visual = nn.Linear(self.hidden_dim + self.in_channels, self.hidden_dim)
        self.lin_edge_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)

        self.context_obj = SHA_Encoder(config, self.obj_layer)
        self.context_edge = SHA_Decoder(config, self.edge_layer)
        # self.context_edge = SHA_Encoder(config, self.edge_layer)

    def forward(self, roi_features, proposals, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer

        num_objs = [len(p) for p in proposals]
        obj_pre_rep_vis = cat((roi_features, pos_embed), -1)
        obj_pre_rep_vis = self.lin_obj_visual(obj_pre_rep_vis)
        obj_pre_rep_txt = obj_embed
        obj_pre_rep_txt = self.lin_obj_textual(obj_pre_rep_txt)
        obj_feats_vis, _, = self.context_obj(obj_pre_rep_vis, obj_pre_rep_txt, num_objs)
        obj_feats = obj_feats_vis

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            # edge_pre_rep_vis = cat((obj_feats, pos_embed), -1)
            edge_pre_rep_txt = self.obj_embed2(obj_labels)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            # edge_pre_rep_vis = obj_feats
            edge_pre_rep_txt = self.obj_embed2(obj_preds)

        # edge context
        edge_pre_rep_vis = self.lin_edge_visual(edge_pre_rep_vis)
        edge_pre_rep_txt = self.lin_edge_textual(edge_pre_rep_txt)
        edge_ctx_vis, _ = self.context_edge(edge_pre_rep_vis, edge_pre_rep_txt, num_objs)
        edge_ctx = edge_ctx_vis

        return obj_dists, obj_preds, edge_ctx

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

# we rearrange the VG dataset, sort the relation classes in descending order (the original order is based on relation class names)
predicate_new_order = [0, 10, 42, 43, 34, 28, 17, 19, 7, 29, 33, 18, 35, 32, 27, 50, 22, 44, 45, 25, 2, 9, 5, 15, 26, 23, 37, 48, 41, 6, 4, 1, 38, 21, 46, 30, 36, 47, 14, 49, 11, 16, 39, 13, 31, 40, 20, 24, 3, 12, 8]
predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712, 5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352, 663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270, 234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
predicate_new_order_name = ['__background__', 'on', 'has', 'wearing', 'of', 'in', 'near', 'behind', 'with', 'holding', 'above', 'sitting on', 'wears', 'under', 'riding', 'in front of', 'standing on', 'at', 'carrying', 'attached to', 'walking on', 'over', 'for', 'looking at', 'watching', 'hanging from', 'laying on', 'eating', 'and', 'belonging to', 'parked on', 'using', 'covering', 'between', 'along', 'covered in', 'part of', 'lying on', 'on back of', 'to', 'walking in', 'mounted on', 'across', 'against', 'from', 'growing on', 'painted on', 'playing', 'made of', 'says', 'flying in']

def get_group_splits(Dataset_name, split_name):
    assert Dataset_name in ['VG', 'GQA_200']
    incremental_stage_list = None
    predicate_stage_count = None
    if Dataset_name == 'VG':
        assert split_name in ['divide3', 'divide4', 'divide5', 'average']
        if split_name == 'divide3':#[]
            incremental_stage_list = [[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9, 10, 11, 12, 13, 14],
                                      [15, 16, 17, 18, 19, 20],
                                      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
            predicate_stage_count = [3, 3, 8, 6, 20, 10]
        elif split_name == 'divide4':#[4,4,9,19,12]
            incremental_stage_list = [[1, 2, 3, 4],
                                      [5, 6, 7, 8, 9, 10],
                                      [11, 12, 13, 14, 15, 16, 17, 18, 19],
                                      [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
                                      [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
            predicate_stage_count = [4, 6, 9, 19, 12]
        elif split_name == 'divide5':
            incremental_stage_list = [[1, 2, 3, 4],
                                      [5, 6, 7, 8, 9, 10, 11, 12],
                                      [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                      [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
            predicate_stage_count = [4, 8, 10, 28]
        elif split_name == 'average':
            incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                                      [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
            predicate_stage_count = [10, 10, 10, 10, 10]
        else:
            exit('wrong mode in group split!')
        assert sum(predicate_stage_count) == 50

    elif Dataset_name == 'GQA_200':
        assert split_name in ['divide3', 'divide4', 'divide5', 'average']
        if split_name == 'divide3':  # []
            incremental_stage_list = [[1, 2, 3, 4],
                                      [5, 6, 7, 8],
                                      [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                      [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                                      [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [4, 4, 11, 16, 31, 34]
        elif split_name == 'divide4':  # [4,4,9,19,12]
            incremental_stage_list = [[1, 2, 3, 4, 5],
                                      [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                      [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [5, 10, 20, 65]
        elif split_name == 'divide5':
            incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7],
                                      [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                                      [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                                      [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [7, 14, 28, 51]
        elif split_name == 'average':
            incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                                      [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
                                      [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [20, 20, 20, 20, 20]
        else:
            exit('wrong mode in group split!')
        assert sum(predicate_stage_count) == 100

    else:
        exit('wrong mode in group split!')

    return incremental_stage_list, predicate_stage_count

import numpy as np

def generate_current_predicate_set(incremental_stage_list, current_training_stage):
    outp = []
    formerp = []
    current_chosen_vector = []
    former_chosen_vector = []
    for i in range(current_training_stage + 1):
        outp.extend(incremental_stage_list[i])
    for i in range(current_training_stage):
        formerp.extend(incremental_stage_list[i])
    for i in range(len(outp)+1):
        if i in incremental_stage_list[current_training_stage]:
            current_chosen_vector.append(1)
        else:
            current_chosen_vector.append(0)
    for i in range(len(outp)+1):
        if i in formerp:
            former_chosen_vector.append(1)
        else:
            former_chosen_vector.append(0)
    num_stage_vector = []
    n_p = 0
    for isl in incremental_stage_list:
        n_p += len(isl)
        num_stage_vector.append(n_p)

    return outp, formerp, current_chosen_vector, former_chosen_vector, num_stage_vector

def generate_num_stage_vector(incremental_stage_list):
    num_stage_vector = []
    n_p = 0
    for isl in incremental_stage_list:
        n_p += len(isl)
        num_stage_vector.append(n_p)

    return num_stage_vector

def get_current_predicate_idx(incremental_stage_list, zeros_vector_penalty, Dataset_choice):
    data_long = 0
    if Dataset_choice == 'VG':
        data_long = 51
    elif Dataset_choice == 'GQA_200':
        data_long = 101
    else:
        exit('wrong mode in Dataset choice')
    outp = []
    for i in range(data_long):
        outp.append(0)
    for i in range(len(incremental_stage_list)):
        for num in incremental_stage_list[i]:
            outp[num] = i+1
    max_p = []
    for i in incremental_stage_list:
        max_p.append(max(i))

    idx_search_p = []
    kd_p = []
    for i in range(len(incremental_stage_list)):
        p1 = []
        p2 = []
        for j in range(data_long):
            p1.append(0)
            p2.append(zeros_vector_penalty)
        max_l = max_p[i]
        for j in range(max_l):
            p1[j+1] = j+1
            p2[j+1] = 1.0
        idx_search_p.append(p1)
        kd_p.append(p2)

    # for i in idx_search_p:
    #     print(i)
    # print()
    # for i in kd_p:
    #     print(i)
    return outp, max_p, idx_search_p, kd_p

def generate_onehot_vector(incremental_stage_list, current_training_stage, Dataset_choice):
    data_long = 0
    if Dataset_choice == 'VG':
        data_long = 51
    elif Dataset_choice == 'GQA_200':
        data_long = 101
    else:
        exit('wrong mode in Dataset choice')
    one_hot_vector = []
    if current_training_stage == -1:
        one_hot_vector.append(0)
        for i in range(data_long-1):
            one_hot_vector.append(1)
        return one_hot_vector
    for i in range(data_long):
        one_hot_vector.append(0)
    for i in range(current_training_stage+1):
        if i+1 == current_training_stage:
            for idx in incremental_stage_list[i]:
                if idx != 1 and idx != 2:
                    one_hot_vector[idx] = 1
                else:
                    one_hot_vector[idx] = -1
        elif i == current_training_stage:
            for idx in incremental_stage_list[i]:
                one_hot_vector[idx] = 1
        else:
            for idx in incremental_stage_list[i]:
                one_hot_vector[idx] = -1

    return one_hot_vector

def generate_sample_rate_vector(Dataset_choice, num_stage_predicate):
    if Dataset_choice == 'VG':
        predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712,
                                     5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352,
                                     663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270,
                                     234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
        assert len(predicate_new_order_count) == 51
    elif Dataset_choice == 'GQA_200':
        predicate_new_order_count = [200000, 64218, 47205, 32126, 25203, 21104, 15890, 15676, 7688, 6966, 6596, 6044, 5250, 4260, 4180, 4131, 2859, 2559, 2368, 2351, 2134, 1673, 1532, 1373, 1273, 1175, 1139, 1123, 1077, 941, 916, 849, 835, 808, 782, 767, 628, 603, 569, 540, 494, 416, 412, 412, 398, 395, 394, 390, 345, 327, 302, 301, 292, 275, 270, 267, 267, 264, 258, 251, 233, 233, 229, 224, 215, 214, 209, 204, 198, 195, 192, 191, 185, 181, 176, 158, 158, 154, 151, 148, 143, 136, 131, 130, 130, 128, 127, 125, 124, 124, 121, 118, 112, 112, 106, 105, 104, 103, 102, 52, 52]
        assert len(predicate_new_order_count) == 101
    else:
        exit('wrong mode in Dataset_choice')
    outp = []
    for i in range(len(num_stage_predicate)):
        opiece = []
        for j in range(len(predicate_new_order_count)):
            opiece.append(0.0)
        num_list = predicate_new_order_count[0:(num_stage_predicate[i]+1)]
        median = np.median(num_list[1:])
        for j in range(len(num_list)):
            if num_list[j] > median:
                num = median / num_list[j]
                if j == 0:
                    num = num * 10.0
                if num < 0.01:
                    num = 0.01
                opiece[j] = num
            else:
                opiece[j] = 1.0
        outp.append(opiece)
    return outp

def generate_current_group_sequence_for_bias(current_set, Dataset_choice):
    data_long = 0
    if Dataset_choice == 'VG':
        data_long = 51
    elif Dataset_choice == 'GQA_200':
        data_long = 101
    else:
        exit('wrong mode in Dataset choice')
    outp = []
    for i in range(data_long):
        outp.append(0)
    for i in current_set:
        outp[i] = i
    return outp

def generate_current_sequence_for_bias(incremental_stage_list, Dataset_choice):
    data_long = 0
    if Dataset_choice == 'VG':
        data_long = 51
    elif Dataset_choice == 'GQA_200':
        data_long = 101
    else:
        exit('wrong mode in Dataset choice')
    outp = []
    for i in range(len(incremental_stage_list)):
        opiece = []
        for j in range(data_long):
            opiece.append(0)
        for j in range(i+1):
            for k in incremental_stage_list[j]:
                opiece[k] = k
        outp.append(opiece)

    return outp

'-----------------------------------------------------------------------------------'
@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.devices = config.MODEL.DEVICE
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        # use frequence bias
        if self.use_bias:
            freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
            freq_dists_bias = F.dropout(freq_dists_bias, 0.3, training=self.training)
            rel_dists = rel_dists + freq_dists_bias


        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("TransformerSuperPredictor")
class TransformerSuperPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerSuperPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)

        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
        self.devices = config.MODEL.DEVICE
        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        # Clean Head
        self.val_alpha = config.MODEL.ROI_RELATION_HEAD.VAL_ALPHA
        self.rel_compress_clean = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress_clean = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
        layer_init(self.rel_compress_clean, xavier=True)
        layer_init(self.ctx_compress_clean, xavier=True)

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias_clean = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists_noisy = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        # use frequence bias
        if self.use_bias:
            freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
            freq_dists_bias = F.dropout(freq_dists_bias, 0.3, training=self.training)
            rel_dists_noisy = rel_dists_noisy + freq_dists_bias

        rel_dists_clean = self.rel_compress_clean(visual_rep) + self.ctx_compress_clean(prod_rep)
        if self.use_bias:
            freq_dists_bias_clean = self.freq_bias_clean.index_with_labels(pair_pred)
            freq_dists_bias_clean = F.dropout(freq_dists_bias_clean, 0.3, training=self.training)
            rel_dists_clean = rel_dists_clean + freq_dists_bias_clean

        rel_dists = rel_dists_clean
        if not self.training:
            rel_dists = (1.0 - self.val_alpha) * rel_dists + self.val_alpha * rel_dists_noisy

        add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("TransLike_GCL")
class TransLike_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(TransLike_GCL, self).__init__()
        # load parameters
        # import pdb
        # pdb.set_trace()
        self.config = config
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels
        # module construct
        self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        # self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        # self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        # self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        # self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        self.no_relation_restrain = True
        self.zero_label_padding_mode = 'rand_insert'
        self.knowledge_loss_coefficient = 1.0
        # generate the auxiliary lists
        self.group_split_mode = 'divide4'
        num_of_group_element_list, predicate_stage_count = get_group_splits('VG', self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, 'VG')
        self.sample_rate_matrix = generate_sample_rate_vector('VG', self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, 'VG')

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        # if self.Knowledge_Transfer_Mode != 'None':
        self.NLL_Loss = nn.NLLLoss()
        self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
        self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
        self.criterion_loss = nn.CrossEntropyLoss()
        # import pdb
        # pdb.set_trace()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        add_losses = {}
        # import pdb
        # pdb.set_trace()
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)
        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.union_single_not_match:
            visual_rep = ctx_gate * self.up_dim(union_features)
        else:
            visual_rep = ctx_gate * union_features

        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)

            num_groups = self.incre_idx_list[max_label.item()]
            if num_groups == 0:
                num_groups = max(self.incre_idx_list)
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])
            # import pdb
            # pdb.set_trace()
            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    rel_idx = self.incre_idx_list[rel_tar]
                    random_num = random.random()
                    for j in range(num_groups):
                        act_idx = num_groups - j
                        threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                        if random_num <= threshold_cur or act_idx < rel_idx:
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break
            # import pdb
            # pdb.set_trace()
            for i in range(num_groups):
                # import pdb
                # pdb.set_trace()
                if max_label == 0:
                    group_visual = visual_rep
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_visual = visual_rep[cur_chosen_matrix[i]]
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy Loss'''
                # import pdb
                # pdb.set_trace()
                jdx = i
                rel_compress_now = self.rel_compress_all[jdx]
                ctx_compress_now = self.ctx_compress_all[jdx]
                group_output_now = rel_compress_now(group_visual) + ctx_compress_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                # import pdb
                # pdb.set_trace()
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

            return None, None, add_losses
        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, 'VG', predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, 'VG', predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, 'VG', predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, 'VG', predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, 'VG', predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, 'VG',
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, 'VG',
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

@registry.ROI_RELATION_PREDICTOR.register("TransformerTransferPredictor")
class TransformerTransferPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerTransferPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.with_knowdist = False
        self.devices = config.MODEL.DEVICE
        self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
        self.with_cleanclf = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']

        assert self.num_obj_cls == len(obj_classes)
        if len(att_classes) > 0:
            assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.val_alpha = config.MODEL.ROI_RELATION_HEAD.VAL_ALPHA

        # module construct
        # self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False
           
        # initialize layer parameters
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        # self.criterion_loss = nn.CrossEntropyLoss()
        # the transfer classifier
        if self.with_cleanclf:
            self.rel_compress_clean = nn.Linear(self.pooling_dim, self.num_rel_cls)
            self.ctx_compress_clean = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
            layer_init(self.rel_compress_clean, xavier=True)
            layer_init(self.ctx_compress_clean, xavier=True)
            self.gcns_rel_clean = GCN(self.pooling_dim, self.pooling_dim, self.dropout_rate)
            self.gcns_ctx_clean = GCN(self.hidden_dim * 2, self.hidden_dim * 2, self.dropout_rate)
            self.freq_bias_clean = FrequencyBias(config, statistics)
        if self.with_transfer:
            #pred_adj_np = np.load('./misc/conf_mat_adj_mat.npy')
            print("Using Confusion Matrix Transfer!")
            pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
            # pred_adj_np = 1.0 - pred_adj_np
            pred_adj_np[0, :] = 0.0
            pred_adj_np[:, 0] = 0.0
            pred_adj_np[0, 0] = 1.0
            # adj_i_j means the baseline outputs category j, but the ground truth is i.
            pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
            pred_adj_np = adj_normalize(pred_adj_np)
            self.pred_adj_nor = torch.from_numpy(pred_adj_np).float().to(self.devices)
            self.pred_adj_layer_clean = nn.Linear(self.num_rel_cls, self.num_rel_cls, bias=False)
            layer_init(self.pred_adj_layer_clean, xavier=True)
            with torch.no_grad():
                self.pred_adj_layer_clean.weight.copy_(torch.eye(self.num_rel_cls,dtype=torch.float), non_blocking=True)
                self.pred_adj_layer_clean.weight.copy_(self.pred_adj_nor, non_blocking=True)
        self.loss_evaluator = make_roi_relation_loss_evaluator(config)

                
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            # import pdb
            # pdb.set_trace()
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists_general = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        if self.use_bias:
            freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
            freq_dists_bias = F.dropout(freq_dists_bias, 0.3, training=self.training)
            rel_dists_general = rel_dists_general + freq_dists_bias
        rel_dists = rel_dists_general
        # # the transfer classifier
        if self.with_cleanclf:
            rel_dists_clean = self.rel_compress_clean(visual_rep) + self.ctx_compress_clean(prod_rep)
            if self.use_bias:
                freq_dists_bias_clean = self.freq_bias_clean.index_with_labels(pair_pred)
                freq_dists_bias_clean = F.dropout(freq_dists_bias_clean, 0.3, training=self.training)
                rel_dists_clean = rel_dists_clean + freq_dists_bias_clean
            rel_dists = rel_dists_clean

        if self.with_transfer:
            rel_dists = (self.pred_adj_nor @ rel_dists.T).T

        add_losses = {}
        # if self.with_knowdist:
        #     rel_dists_specific_soft = F.log_softmax(rel_dists, -1)
        #     rel_dists_general_soft = F.softmax(rel_dists_general, -1)
        #     add_losses['know_dist_kl'] = self.kd_alpha * self.kl_loss(rel_dists_specific_soft, rel_dists_general_soft)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        # if self.training:
        #     loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, rel_dists, obj_dists)
        #     output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
        #     return obj_dists, rel_dists, output_losses
        # else:
        #     return obj_dists, rel_dists, add_losses

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses

# 提点
# @registry.ROI_RELATION_PREDICTOR.register("TransformerTransferPredictor")
# class TransformerTransferPredictor(nn.Module):
#     def __init__(self, config, in_channels):
#         super(TransformerTransferPredictor, self).__init__()
#         self.attribute_on = config.MODEL.ATTRIBUTE_ON
#         # load parameters
#         self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
#         self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
#         self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
#         self.dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE

#         assert in_channels is not None
#         num_inputs = in_channels
#         self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
#         self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
#         self.with_knowdist = False
#         self.devices = config.MODEL.DEVICE
#         self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
#         self.with_cleanclf = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
#         # load class dict
#         statistics = get_dataset_statistics(config)
#         obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
#             'att_classes']
#         assert self.num_obj_cls == len(obj_classes)
#         assert self.num_att_cls == len(att_classes)
#         assert self.num_rel_cls == len(rel_classes)
#         self.val_alpha = config.MODEL.ROI_RELATION_HEAD.VAL_ALPHA

#         # module construct
#         # self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
#         self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

#         # post decoding
#         self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
#         self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
#         self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
#         layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
#         self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
#         layer_init(self.post_cat, xavier=True)
        
#         if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
#             self.union_single_not_match = True
#             self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
#             layer_init(self.up_dim, xavier=True)
#         else:
#             self.union_single_not_match = False
           
#         # initialize layer parameters
#         self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
#         self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
#         layer_init(self.rel_compress, xavier=True)
#         layer_init(self.ctx_compress, xavier=True)
#         if self.use_bias:
#             # convey statistics into FrequencyBias to avoid loading again
#             self.freq_bias = FrequencyBias(config, statistics)
#         self.criterion_loss = nn.CrossEntropyLoss()
#         # the transfer classifier
#         if self.with_cleanclf:
#             self.rel_compress_clean = nn.Linear(self.pooling_dim, self.num_rel_cls)
#             self.ctx_compress_clean = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
#             layer_init(self.rel_compress_clean, xavier=True)
#             layer_init(self.ctx_compress_clean, xavier=True)
#             self.gcns_rel_clean = GCN(self.pooling_dim, self.pooling_dim, self.dropout_rate)
#             self.gcns_ctx_clean = GCN(self.hidden_dim * 2, self.hidden_dim * 2, self.dropout_rate)
#             self.freq_bias_clean = FrequencyBias(config, statistics)
#         if self.with_transfer:
#             #pred_adj_np = np.load('./misc/conf_mat_adj_mat.npy')
#             print("Using Confusion Matrix Transfer!")
#             pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
#             # pred_adj_np = 1.0 - pred_adj_np
#             pred_adj_np[0, :] = 0.0
#             pred_adj_np[:, 0] = 0.0
#             pred_adj_np[0, 0] = 1.0
#             # adj_i_j means the baseline outputs category j, but the ground truth is i.
#             pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
#             pred_adj_np = adj_normalize(pred_adj_np)
#             self.pred_adj_nor = torch.from_numpy(pred_adj_np).float().to(self.devices)
#             self.pred_adj_layer_clean = nn.Linear(self.num_rel_cls, self.num_rel_cls, bias=False)
#             layer_init(self.pred_adj_layer_clean, xavier=True)
#             with torch.no_grad():
#                 self.pred_adj_layer_clean.weight.copy_(torch.eye(self.num_rel_cls,dtype=torch.float), non_blocking=True)
#                 self.pred_adj_layer_clean.weight.copy_(self.pred_adj_nor, non_blocking=True)
#         self.loss_evaluator = make_roi_relation_loss_evaluator(config)

                
#     def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
#         """
#         Returns:
#             obj_dists (list[Tensor]): logits of object label distribution
#             rel_dists (list[Tensor])
#             rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
#             union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
#         """
#         if self.attribute_on:
#             obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
#         else:
#             # import pdb
#             # pdb.set_trace()
#             obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

#         # post decode
#         edge_rep = self.post_emb(edge_ctx)
#         edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
#         head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
#         tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

#         num_rels = [r.shape[0] for r in rel_pair_idxs]
#         num_objs = [len(b) for b in proposals]
#         assert len(num_rels) == len(num_objs)

#         head_reps = head_rep.split(num_objs, dim=0)
#         tail_reps = tail_rep.split(num_objs, dim=0)
#         obj_preds = obj_preds.split(num_objs, dim=0)

#         # from object level feature to pairwise relation level feature
#         prod_reps = []
#         pair_preds = []
#         for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
#             prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
#             pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
#         prod_rep = cat(prod_reps, dim=0)
#         pair_pred = cat(pair_preds, dim=0)

#         ctx_gate = self.post_cat(prod_rep)

#         # use union box and mask convolution
#         if self.use_vision:
#             if self.union_single_not_match:
#                 visual_rep = ctx_gate * self.up_dim(union_features)
#             else:
#                 visual_rep = ctx_gate * union_features

#         rel_dists_general = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
#         if self.use_bias:
#             freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
#             freq_dists_bias = F.dropout(freq_dists_bias, 0.3, training=self.training)
#             rel_dists_general = rel_dists_general + freq_dists_bias
#         rel_dists = rel_dists_general
#         # # the transfer classifier
#         if self.with_cleanclf:
#             rel_dists_clean = self.rel_compress_clean(visual_rep) + self.ctx_compress_clean(prod_rep)
#             if self.use_bias:
#                 freq_dists_bias_clean = self.freq_bias_clean.index_with_labels(pair_pred)
#                 freq_dists_bias_clean = F.dropout(freq_dists_bias_clean, 0.3, training=self.training)
#                 rel_dists_clean = rel_dists_clean + freq_dists_bias_clean
#             rel_dists = rel_dists_clean

#         if self.with_transfer:
#             rel_dists = (self.pred_adj_nor @ rel_dists.T).T

#         add_losses = {}
#         if self.with_knowdist:
#             rel_dists_specific_soft = F.log_softmax(rel_dists, -1)
#             rel_dists_general_soft = F.softmax(rel_dists_general, -1)
#             add_losses['know_dist_kl'] = self.kd_alpha * self.kl_loss(rel_dists_specific_soft, rel_dists_general_soft)

#         obj_dists = obj_dists.split(num_objs, dim=0)
#         rel_dists = rel_dists.split(num_rels, dim=0)
#         if self.training:
#             loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, rel_dists, obj_dists)
#             output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
#             return obj_dists, rel_dists, output_losses
#         else:
#             return obj_dists, rel_dists, add_losses

#         if self.attribute_on:
#             att_dists = att_dists.split(num_objs, dim=0)
#             return (obj_dists, att_dists), rel_dists, add_losses
#         else:
#             return obj_dists, rel_dists, output_losses

# @registry.ROI_RELATION_PREDICTOR.register("TransformerTransferPredictor")
# class TransformerTransferPredictor(nn.Module):
#     def __init__(self, config, in_channels):
#         super(TransformerTransferPredictor, self).__init__()
#         self.attribute_on = config.MODEL.ATTRIBUTE_ON
#         # load parameters
#         self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
#         self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
#         self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
#         self.dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE

#         assert in_channels is not None
#         num_inputs = in_channels
#         self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
#         self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
#         self.with_knowdist = False
#         self.devices = config.MODEL.DEVICE
#         self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
#         self.with_cleanclf = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
#         # load class dict
#         statistics = get_dataset_statistics(config)
#         obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
#             'att_classes']
#         assert self.num_obj_cls == len(obj_classes)
#         assert self.num_att_cls == len(att_classes)
#         assert self.num_rel_cls == len(rel_classes)
#         self.val_alpha = config.MODEL.ROI_RELATION_HEAD.VAL_ALPHA

#         # module construct
#         # self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
#         self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

#         # post decoding
#         self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
#         self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
#         self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
#         layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
#         self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
#         layer_init(self.post_cat, xavier=True)
        
#         if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
#             self.union_single_not_match = True
#             self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
#             layer_init(self.up_dim, xavier=True)
#         else:
#             self.union_single_not_match = False
           
#         # initialize layer parameters
#         self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
#         self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
#         layer_init(self.rel_compress, xavier=True)
#         layer_init(self.ctx_compress, xavier=True)
#         if self.use_bias:
#             # convey statistics into FrequencyBias to avoid loading again
#             self.freq_bias = FrequencyBias(config, statistics)

#         # the transfer classifier
#         if self.with_cleanclf:
#             self.rel_compress_clean = nn.Linear(self.pooling_dim, self.num_rel_cls)
#             self.ctx_compress_clean = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
#             layer_init(self.rel_compress_clean, xavier=True)
#             layer_init(self.ctx_compress_clean, xavier=True)
#             # self.gcns_rel_clean = GCN(self.pooling_dim, self.pooling_dim, self.dropout_rate)
#             # self.gcns_ctx_clean = GCN(self.hidden_dim * 2, self.hidden_dim * 2, self.dropout_rate)
#             self.freq_bias_clean = FrequencyBias(config, statistics)
#         if self.with_transfer:
#             #pred_adj_np = np.load('./misc/conf_mat_adj_mat.npy')
#             print("Using Confusion Matrix Transfer!")
#             pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
#             # pred_adj_np = 1.0 - pred_adj_np
#             pred_adj_np[0, :] = 0.0
#             pred_adj_np[:, 0] = 0.0
#             pred_adj_np[0, 0] = 1.0
#             # adj_i_j means the baseline outputs category j, but the ground truth is i.
#             pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
#             pred_adj_np = adj_normalize(pred_adj_np)
#             self.pred_adj_nor = torch.from_numpy(pred_adj_np).float().to(self.devices)
#             # self.pred_adj_layer_clean = nn.Linear(self.num_rel_cls, self.num_rel_cls, bias=False)
#             # #layer_init(self.pred_adj_layer_clean, xavier=True)
#             # with torch.no_grad():
#             #     self.pred_adj_layer_clean.weight.copy_(torch.eye(self.num_rel_cls,dtype=torch.float), non_blocking=True)
#                 #self.pred_adj_layer_clean.weight.copy_(self.pred_adj_nor, non_blocking=True)
                
#     def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
#         """
#         Returns:
#             obj_dists (list[Tensor]): logits of object label distribution
#             rel_dists (list[Tensor])
#             rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
#             union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
#         """
#         if self.attribute_on:
#             obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
#         else:
#             # import pdb
#             # pdb.set_trace()
#             obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

#         # post decode
#         edge_rep = self.post_emb(edge_ctx)
#         edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
#         head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
#         tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

#         num_rels = [r.shape[0] for r in rel_pair_idxs]
#         num_objs = [len(b) for b in proposals]
#         assert len(num_rels) == len(num_objs)

#         head_reps = head_rep.split(num_objs, dim=0)
#         tail_reps = tail_rep.split(num_objs, dim=0)
#         obj_preds = obj_preds.split(num_objs, dim=0)

#         # from object level feature to pairwise relation level feature
#         prod_reps = []
#         pair_preds = []
#         for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
#             prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
#             pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
#         prod_rep = cat(prod_reps, dim=0)
#         pair_pred = cat(pair_preds, dim=0)

#         ctx_gate = self.post_cat(prod_rep)

#         # use union box and mask convolution
#         if self.use_vision:
#             if self.union_single_not_match:
#                 visual_rep = ctx_gate * self.up_dim(union_features)
#             else:
#                 visual_rep = ctx_gate * union_features

#         rel_dists_general = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
#         if self.use_bias:
#             freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
#             freq_dists_bias = F.dropout(freq_dists_bias, 0.3, training=self.training)
#             rel_dists_general = rel_dists_general + freq_dists_bias
#         rel_dists = rel_dists_general
#         # the transfer classifier
#         if self.with_cleanclf:
#             rel_dists_clean = self.rel_compress_clean(visual_rep) + self.ctx_compress_clean(prod_rep)
#             if self.use_bias:
#                 freq_dists_bias_clean = self.freq_bias_clean.index_with_labels(pair_pred)
#                 freq_dists_bias_clean = F.dropout(freq_dists_bias_clean, 0.3, training=self.training)
#                 rel_dists_clean = rel_dists_clean + freq_dists_bias_clean
#             rel_dists = rel_dists_clean

#         if self.with_transfer:
#             rel_dists = (self.pred_adj_nor @ rel_dists.T).T

#         add_losses = {}
#         # if self.with_knowdist:
#         #     rel_dists_specific_soft = F.log_softmax(rel_dists, -1)
#         #     rel_dists_general_soft = F.softmax(rel_dists_general, -1)
#         #     add_losses['know_dist_kl'] = self.kd_alpha * self.kl_loss(rel_dists_specific_soft, rel_dists_general_soft)

#         obj_dists = obj_dists.split(num_objs, dim=0)
#         rel_dists = rel_dists.split(num_rels, dim=0)

#         if self.attribute_on:
#             att_dists = att_dists.split(num_objs, dim=0)
#             return (obj_dists, att_dists), rel_dists, add_losses
#         else:
#             return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.with_clean_classifier = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER

        if self.with_clean_classifier:
            if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
                self.union_single_not_match = True
                self.up_dim_clean = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
                layer_init(self.up_dim_clean, xavier=True)
            else:
                self.union_single_not_match = False
            self.post_cat_clean = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
            self.rel_compress_clean = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
            layer_init(self.post_cat_clean, xavier=True)
            layer_init(self.rel_compress_clean, xavier=True)
            if self.use_bias:
                # convey statistics into FrequencyBias to avoid loading again
                self.freq_bias_clean = FrequencyBias(config, statistics)
            self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
            if self.with_transfer:
                self.devices = config.MODEL.DEVICE
                print("!!!!!!!!!With Confusion Matrix Channel!!!!!")
                pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
                # pred_adj_np = 1.0 - pred_adj_np
                pred_adj_np[0, :] = 0.0
                pred_adj_np[:, 0] = 0.0
                pred_adj_np[0, 0] = 1.0
                # adj_i_j means the baseline outputs category j, but the ground truth is i.
                pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
                pred_adj_np = adj_normalize(pred_adj_np)
                self.pred_adj_nor = torch.from_numpy(pred_adj_np).float().to(self.devices)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            freq_dists_bias = self.freq_bias.index_with_labels(pair_pred.long())
            freq_dists_bias = F.dropout(freq_dists_bias, 0.3, training=self.training)
            rel_dists = rel_dists + freq_dists_bias

        if self.with_clean_classifier:
            prod_rep_clean = cat(prod_reps, dim=0)
            prod_rep_clean = self.post_cat_clean(prod_rep_clean)
            if self.use_vision:
                if self.union_single_not_match:
                    prod_rep_clean = prod_rep_clean * self.up_dim_clean(union_features)
                else:
                    prod_rep_clean = prod_rep_clean * union_features

            rel_dists_clean = self.rel_compress_clean(prod_rep_clean)
            if self.use_bias:
                freq_dists_bias_clean = self.freq_bias_clean.index_with_labels(pair_pred.long())
                freq_dists_bias_clean = F.dropout(freq_dists_bias_clean, 0.3, training=self.training)
                rel_dists_clean = rel_dists_clean + freq_dists_bias_clean
            if self.with_transfer:
                rel_dists_clean = (self.pred_adj_nor @ rel_dists_clean.T).T

            rel_dists = rel_dists_clean

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)


        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses

# @registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
# class MotifPredictor(nn.Module):
#     def __init__(self, config, in_channels):
#         super(MotifPredictor, self).__init__()
#         self.attribute_on = config.MODEL.ATTRIBUTE_ON
#         self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
#         self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
#         self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

#         assert in_channels is not None
#         num_inputs = in_channels
#         self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
#         self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

#         # load class dict
#         statistics = get_dataset_statistics(config)
#         obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
#             'att_classes']
#         assert self.num_obj_cls == len(obj_classes)
#         assert self.num_att_cls == len(att_classes)
#         assert self.num_rel_cls == len(rel_classes)
#         # init contextual lstm encoding
#         if self.attribute_on:
#             self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
#         else:
#             self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

#         # post decoding
#         self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
#         self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
#         self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
#         self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
#         self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

#         # initialize layer parameters
#         layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
#         layer_init(self.post_cat, xavier=True)
#         layer_init(self.rel_compress, xavier=True)

#         if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
#             self.union_single_not_match = True
#             self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
#             layer_init(self.up_dim, xavier=True)
#         else:
#             self.union_single_not_match = False

#         if self.use_bias:
#             # convey statistics into FrequencyBias to avoid loading again
#             self.freq_bias = FrequencyBias(config, statistics)

#         self.with_clean_classifier = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER

#         if self.with_clean_classifier:
#             if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
#                 self.union_single_not_match = True
#                 self.up_dim_clean = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
#                 layer_init(self.up_dim_clean, xavier=True)
#             else:
#                 self.union_single_not_match = False
#             self.post_cat_clean = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
#             self.rel_compress_clean = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
#             layer_init(self.post_cat_clean, xavier=True)
#             layer_init(self.rel_compress_clean, xavier=True)
#             if self.use_bias:
#                 # convey statistics into FrequencyBias to avoid loading again
#                 self.freq_bias_clean = FrequencyBias(config, statistics)
#             self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
#             if self.with_transfer:
#                 self.devices = config.MODEL.DEVICE
#                 print("!!!!!!!!!With Confusion Matrix Channel!!!!!")
#                 pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
#                 # pred_adj_np = 1.0 - pred_adj_np
#                 pred_adj_np[0, :] = 0.0
#                 pred_adj_np[:, 0] = 0.0
#                 pred_adj_np[0, 0] = 1.0
#                 # adj_i_j means the baseline outputs category j, but the ground truth is i.
#                 pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
#                 pred_adj_np = adj_normalize(pred_adj_np)
#                 self.pred_adj_nor = torch.from_numpy(pred_adj_np).float().to(self.devices)

#     def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
#         """
#         Returns:
#             obj_dists (list[Tensor]): logits of object label distribution
#             rel_dists (list[Tensor])
#             rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
#             union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
#         """

#         # encode context infomation
#         if self.attribute_on:
#             obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
#         else:
#             obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

#         # post decode
#         edge_rep = self.post_emb(edge_ctx)
#         edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
#         head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
#         tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

#         num_rels = [r.shape[0] for r in rel_pair_idxs]
#         num_objs = [len(b) for b in proposals]
#         assert len(num_rels) == len(num_objs)

#         head_reps = head_rep.split(num_objs, dim=0)
#         tail_reps = tail_rep.split(num_objs, dim=0)
#         obj_preds = obj_preds.split(num_objs, dim=0)

#         prod_reps = []
#         pair_preds = []
#         for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
#             prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
#             pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
#         prod_rep = cat(prod_reps, dim=0)
#         pair_pred = cat(pair_preds, dim=0)
#         prod_rep = self.post_cat(prod_rep)

#         if self.use_vision:
#             if self.union_single_not_match:
#                 prod_rep = prod_rep * self.up_dim(union_features)
#             else:
#                 prod_rep = prod_rep * union_features

#         rel_dists = self.rel_compress(prod_rep)

#         if self.use_bias:
#             freq_dists_bias = self.freq_bias.index_with_labels(pair_pred.long())
#             freq_dists_bias = F.dropout(freq_dists_bias, 0.3, training=self.training)
#             rel_dists = rel_dists + freq_dists_bias

#         if self.with_clean_classifier:
#             prod_rep_clean = cat(prod_reps, dim=0)
#             prod_rep_clean = self.post_cat_clean(prod_rep_clean)
#             if self.use_vision:
#                 if self.union_single_not_match:
#                     prod_rep_clean = prod_rep_clean * self.up_dim_clean(union_features)
#                 else:
#                     prod_rep_clean = prod_rep_clean * union_features

#             rel_dists_clean = self.rel_compress_clean(prod_rep_clean)
#             if self.use_bias:
#                 freq_dists_bias_clean = self.freq_bias_clean.index_with_labels(pair_pred.long())
#                 freq_dists_bias_clean = F.dropout(freq_dists_bias_clean, 0.3, training=self.training)
#                 rel_dists_clean = rel_dists_clean + freq_dists_bias_clean
#             if self.with_transfer:
#                 rel_dists_clean = (self.pred_adj_nor @ rel_dists_clean.T).T

#             rel_dists = rel_dists_clean

#         obj_dists = obj_dists.split(num_objs, dim=0)
#         rel_dists = rel_dists.split(num_rels, dim=0)


#         # we use obj_preds instead of pred from obj_dists
#         # because in decoder_rnn, preds has been through a nms stage
#         add_losses = {}

#         if self.attribute_on:
#             att_dists = att_dists.split(num_objs, dim=0)
#             return (obj_dists, att_dists), rel_dists, add_losses
#         else:
#             return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        # self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # layer_init(self.uni_gate, xavier=True)
        # layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        # layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

        self.with_clean_classifier = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER

        if self.with_clean_classifier:
            if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
                self.union_single_not_match = True
                self.up_dim_clean = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
                layer_init(self.up_dim_clean, xavier=True)
            else:
                self.union_single_not_match = False
            self.post_cat_clean = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
            self.ctx_compress_clean = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
            layer_init(self.post_cat_clean, xavier=True)
            layer_init(self.ctx_compress_clean, xavier=True)
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias_clean = FrequencyBias(config, statistics)

            self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
            if self.with_transfer:
                self.devices = config.MODEL.DEVICE
                print("!!!!!!!!!With Confusion Matrix Channel!!!!!")
                pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
                # pred_adj_np = 1.0 - pred_adj_np
                pred_adj_np[0, :] = 0.0
                pred_adj_np[:, 0] = 0.0
                pred_adj_np[0, 0] = 1.0
                # adj_i_j means the baseline outputs category j, but the ground truth is i.
                pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
                pred_adj_np = adj_normalize(pred_adj_np)
                self.pred_adj_nor = torch.from_numpy(pred_adj_np).float().to(self.devices)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        # uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        # frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        # uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())
        frq_dists = F.dropout(frq_dists, 0.3, training=self.training)
        rel_dists = ctx_dists + frq_dists
        # rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists
        if self.with_clean_classifier:
            prod_rep_clean = cat(prod_reps, dim=0)
            prod_rep_clean = self.post_cat_clean(prod_rep_clean)
            if self.union_single_not_match:
                union_features = self.up_dim_clean(union_features)

            ctx_dists_clean = self.ctx_compress_clean(prod_rep_clean * union_features)
            # uni_dists = self.uni_compress(self.drop(union_features))
            frq_dists_clean = self.freq_bias_clean.index_with_labels(pair_pred.long())
            frq_dists_clean = F.dropout(frq_dists_clean, 0.3, training=self.training)
            rel_dists_clean = ctx_dists_clean + frq_dists_clean
            if self.with_transfer:
                rel_dists_clean = (self.pred_adj_nor @ rel_dists_clean.T).T
            rel_dists = rel_dists_clean
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            if binary_preds[0].requires_grad:
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True), ])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)

        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.hidden_dim, self.pooling_dim),
                                           nn.ReLU(inplace=True)
                                           ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger,
                              ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps,
                                                                             obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
            else:
                ctx_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_obj_probs.append(torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(
            roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features,
                                                                                                   proposals,
                                                                                                   rel_pair_idxs,
                                                                                                   num_objs, obj_boxs,
                                                                                                   logger,
                                                                                                   ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats

        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()),
                                                              rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1,
                                                                                          -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':  # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE':  # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            # union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            # union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            # union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            # union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            # union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            # union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            # union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest

        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
