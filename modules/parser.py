from datautil.dataloader import batch_iter
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from .MST import mst_decode
from log.logger_ import logger


class BiaffineParser(object):
    def __init__(self, parser_model):
        super(BiaffineParser, self).__init__()
        assert isinstance(parser_model, nn.Module)
        self.parser_model = parser_model

    def summary(self):
        logger.info(self.parser_model)

    # 训练一次
    def train_iter(self, train_data, args, vocab, optimizer):
        self.parser_model.train()

        train_loss = 0
        all_arc_acc, all_rel_acc, all_arcs = 0, 0, 0
        start_time = time.time()
        nb_batch = int(np.ceil(len(train_data)/args.batch_size))
        batch_size = args.batch_size // args.update_steps
        for i, batcher in enumerate(batch_iter(train_data, batch_size, vocab)):
            batcher = (x.to(args.device) for x in batcher)
            wd_idx, extwd_idx, tag_idx, true_head_idx, true_rel_idx, non_pad_mask = batcher

            pred_arc_score, pred_rel_score = self.parser_model(wd_idx, extwd_idx, tag_idx, non_pad_mask)

            loss = self.calc_loss(pred_arc_score, pred_rel_score, true_head_idx, true_rel_idx, non_pad_mask)
            if args.update_steps > 1:
                loss = loss / args.update_steps
            loss_val = loss.data.item()
            train_loss += loss_val
            loss.backward()

            arc_acc, rel_acc, total_arcs = self.calc_acc(pred_arc_score, pred_rel_score, true_head_idx, true_rel_idx, non_pad_mask)
            all_arc_acc += arc_acc
            all_rel_acc += rel_acc
            all_arcs += total_arcs

            ARC = all_arc_acc * 100 / all_arcs
            REL = all_rel_acc * 100 / all_arcs
            logger.info('Iter%d ARC: %.3f%%, REL: %.3f%%' % (i + 1, ARC, REL))
            logger.info('time cost: %.2fs, train loss: %.2f' % ((time.time() - start_time), loss_val))

            # 梯度累积，相对于变相增大batch_size，节省存储
            if (i+1) % args.update_steps == 0 or (i == nb_batch-1):
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parser_model.parameters()), max_norm=5.)
                optimizer.step()
                self.parser_model.zero_grad()

        train_loss /= len(train_data)
        ARC = all_arc_acc * 100 / all_arcs
        REL = all_rel_acc * 100 / all_arcs

        return train_loss, ARC, REL

    # 训练多次
    def train(self, train_data, dev_data, test_data, args, vocab):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parser_model.parameters()),
                               lr=args.learning_rate, betas=(args.beta1, args.beta2))
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda t: 0.75 ** (t / 500))
        best_uas = 0
        test_best_uas, test_best_las = 0, 0
        for ep in range(1, 1+args.epoch):
            lr_scheduler.step()
            train_loss, arc, rel = self.train_iter(train_data, args, vocab, optimizer)
            dev_uas, dev_las = self.evaluate(dev_data, args, vocab)
            logger.info('[Epoch %d] train loss: %.3f, lr: %f, ARC: %.3f%%, REL: %.3f%%' % (ep, train_loss, lr_scheduler.get_lr()[0], arc, rel))
            logger.info('Dev data -- UAS: %.3f%%, LAS: %.3f%%, best_UAS: %.3f%%' % (dev_uas, dev_las, best_uas))
            if dev_uas > best_uas:
                best_uas = dev_uas
                test_uas, test_las = self.evaluate(test_data, args, vocab)
                if test_best_uas < test_uas:
                    test_best_uas = test_uas
                if test_best_las < test_las:
                    test_best_las = test_las

                logger.info('Test data -- UAS: %.3f%%, LAS: %.3f%%' % (test_uas, test_las))

        logger.info('Final test performance -- UAS: %.3f%%, LAS: %.3f%%' % (test_best_uas, test_best_las))

    def evaluate(self, test_data, args, vocab):
        self.parser_model.eval()

        all_arc_acc, all_rel_acc, all_arcs = 0, 0, 0
        with torch.no_grad():
            for batcher in batch_iter(test_data, args.batch_size, vocab):
                batcher = (x.to(args.device) for x in batcher)
                wd_idx, extwd_idx, tag_idx, true_head_idx, true_rel_idx, non_pad_mask = batcher

                pred_arc_score, pred_rel_score = self.parser_model(wd_idx, extwd_idx, tag_idx, non_pad_mask)

                arc_acc, rel_acc, total_arcs = self.metric_evaluate(pred_arc_score, pred_rel_score, true_head_idx, true_rel_idx, non_pad_mask)
                all_arc_acc += arc_acc
                all_rel_acc += rel_acc
                all_arcs += total_arcs

        uas = all_arc_acc * 100 / all_arcs
        las = all_rel_acc * 100 / all_arcs
        return uas, las

    def metric_evaluate(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask=None):
        '''
        :param pred_arcs: (bz, seq_len, seq_len)
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_heads: (bz, seq_len)  包含padding
        :param true_rels: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        non_pad_mask = non_pad_mask.byte()
        pred_heads = mst_decode(pred_arcs, non_pad_mask)
        non_pad_mask[:, 0] = 0  # mask out <root>

        pred_heads = pred_heads[non_pad_mask]
        true_heads = true_heads[non_pad_mask]
        pred_head_correct = true_heads.eq(pred_heads)

        pred_rels = pred_rels[non_pad_mask]
        pred_rels = pred_rels[torch.arange(len(pred_rels)), pred_heads].argmax(dim=-1)
        true_rels = true_rels[non_pad_mask]
        pred_rel_correct = true_rels.eq(pred_rels) * pred_head_correct

        arc_acc = pred_head_correct.sum().item()
        rel_acc = pred_rel_correct.sum().item()
        total_arcs = non_pad_mask.sum().item()

        # pred_heads_correct = (pred_heads == true_heads) * non_pad_mask
        # arc_acc = pred_heads_correct.sum().item()
        # total_arcs = non_pad_mask.sum().item()
        # bz, seq_len, seq_len, rel_size = pred_rels.size()
        # pred_rels = pred_rels[torch.arange(bz).unsqueeze(1), torch.arange(seq_len).unsqueeze(0), pred_heads].argmax(dim=2)
        # pred_rels_correct = (pred_rels == true_rels) * pred_heads_correct
        # rel_acc = pred_rels_correct.sum().item()

        return arc_acc, rel_acc, total_arcs

    def calc_loss(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask):
        '''
        :param pred_arcs: (bz, seq_len, seq_len)
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_heads: (bz, seq_len)  包含padding
        :param true_rels: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len) 有效部分mask
        :return:
        '''
        non_pad_mask = non_pad_mask.byte()
        # non_pad_mask[:, 0] = 0  # mask out <root>
        pad_mask = (non_pad_mask == 0)

        bz, seq_len, _ = pred_arcs.size()
        masked_true_heads = true_heads.masked_fill(pad_mask, -1)
        arc_loss = F.cross_entropy(pred_arcs.reshape(bz*seq_len, -1), masked_true_heads.reshape(-1), ignore_index=-1)

        bz, seq_len, seq_len, rel_size = pred_rels.size()

        # out_rels = true_rels.data.new_zeros((bz, seq_len, rel_size), dtype=torch.float32)
        # for i, (rels, heads) in enumerate(zip(pred_rels, true_heads)):
        #     # rels: (seq_len, seq_len, rel_size)   heads: (seq_len, )
        #     rel_probs = []
        #     for t in range(seq_len):
        #         rel_probs.append(rels[t][heads[t].item()])
        #     out_rels[i] = torch.stack(tuple(rel_probs), dim=0)  # (seq_len, rel_size)

        out_rels = pred_rels[torch.arange(bz).unsqueeze(1), torch.arange(seq_len).unsqueeze(0), true_heads]

        true_rels = true_rels.masked_fill(pad_mask, -1)
        # (bz*seq_len, rel_size)  (bz*seq_len, )
        rel_loss = F.cross_entropy(out_rels.reshape(-1, rel_size), true_rels.reshape(-1), ignore_index=-1)

        return arc_loss + rel_loss

    def calc_acc(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask=None):
        '''
        :param pred_arcs: (bz, seq_len, seq_len)
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_heads: (bz, seq_len)  包含padding
        :param true_rels: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        non_pad_mask = non_pad_mask.byte()
        non_pad_mask[:, 0] = 0  # mask out <root>

        bz, seq_len, seq_len, rel_size = pred_rels.size()

        # (bz, seq_len)
        pred_heads = pred_arcs.data.argmax(dim=2)
        pred_heads = pred_heads[non_pad_mask]
        true_arcs = true_heads[non_pad_mask]
        arc_acc = true_arcs.eq(pred_heads).sum().item()

        total_arcs = non_pad_mask.sum().item()

        # (bz, seq_len, rel_size)
        # out_rels = true_rels.data.new_zeros(bz, seq_len, rel_size)
        # for i, (rels, heads) in enumerate(zip(pred_rels, true_heads)):
        #     # rels: (seq_len, seq_len, rel_size)   heads: (seq_len, )
        #     rel_probs = []
        #     for t in range(seq_len):
        #         rel_probs.append(rels[t][heads[t].item()])
        #     out_rels[i] = torch.stack(tuple(rel_probs), dim=0)  # (seq_len, rel_size)

        out_rels = pred_rels[torch.arange(bz).unsqueeze(1), torch.arange(seq_len).unsqueeze(0), true_heads]
        pred_rels = out_rels.argmax(dim=2)
        pred_rels = pred_rels[non_pad_mask]
        true_rels = true_rels[non_pad_mask]
        rel_acc = true_rels.eq(pred_rels).sum().item()

        return arc_acc, rel_acc, total_arcs

    # def calc_loss(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask):
    #     '''
    #     :param pred_arcs: (bz, seq_len, seq_len)
    #     :param pred_rels:  (bz, seq_len, seq_len, rel_size)
    #     :param true_heads: (bz, seq_len)  包含padding
    #     :param true_rels: (bz, seq_len)
    #     :param non_pad_mask: (bz, seq_len) 有效部分mask
    #     :return:
    #     '''
    #     non_pad_mask = non_pad_mask.byte()
    #     non_pad_mask[:, 0] = 0  # mask out <root>
    #
    #     pred_heads = pred_arcs[non_pad_mask]  # (bz, seq_len)
    #     true_heads = true_heads[non_pad_mask]   # (bz, )
    #     pred_rels = pred_rels[non_pad_mask]  # (bz, seq_len, rel_size)
    #     pred_rels = pred_rels[torch.arange(len(pred_rels)), true_heads]  # (bz, rel_size)
    #     true_rels = true_rels[non_pad_mask]     # (bz, )
    #
    #     arc_loss = F.cross_entropy(pred_heads, true_heads)
    #     rel_loss = F.cross_entropy(pred_rels, true_rels)
    #
    #     return arc_loss + rel_loss
    #
    # def calc_acc(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask=None):
    #     '''
    #     :param pred_arcs: (bz, seq_len, seq_len)
    #     :param pred_rels:  (bz, seq_len, seq_len, rel_size)
    #     :param true_heads: (bz, seq_len)  包含padding
    #     :param true_rels: (bz, seq_len)
    #     :param non_pad_mask: (bz, seq_len)
    #     :return:
    #     '''
    #     non_pad_mask = non_pad_mask.byte()
    #     non_pad_mask[:, 0] = 0  # mask out <root>
    #
    #     pred_heads = pred_arcs[non_pad_mask]  # (bz, seq_len)
    #     true_heads = true_heads[non_pad_mask]  # (bz, )
    #     pred_heads = pred_heads.data.argmax(dim=-1)
    #     arc_acc = true_heads.eq(pred_heads).sum().item()
    #     total_arcs = non_pad_mask.sum().item()
    #
    #     pred_rels = pred_rels[non_pad_mask]  # (bz, seq_len, rel_size)
    #     pred_rels = pred_rels[torch.arange(len(pred_rels)), true_heads]  # (bz, rel_size)
    #     pred_rels = pred_rels.data.argmax(dim=-1)
    #     true_rels = true_rels[non_pad_mask]  # (bz, )
    #     rel_acc = true_rels.eq(pred_rels).sum().item()
    #
    #     return arc_acc, rel_acc, total_arcs
