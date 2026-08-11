import math
import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss


class ColbertLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        """

        scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)

        # scores = torch.zeros((query_embeddings.shape[0], doc_embeddings.shape[0]), device=query_embeddings.device)
        # for i in range(query_embeddings.shape[0]):
        #     for j in range(doc_embeddings.shape[0]):
        #         # step 1 - dot product --> (s1,s2)
        #         q2d_scores = torch.matmul(query_embeddings[i], doc_embeddings[j].T)
        #         # step 2 -> max on doc  --> (s1)
        #         q_scores = torch.max(q2d_scores, dim=1)[0]
        #         # step 3 --> sum the max score --> (1)
        #         sum_q_score = torch.sum(q_scores)
        #         # step 4 --> assert is scalar
        #         scores[i, j] = sum_q_score

        # assert (scores_einsum - scores < 0.0001).all().item()

        loss_rowwise = self.ce_loss(scores, torch.arange(scores.shape[0], device=scores.device))
        # TODO: comparing between queries might not make sense since it's a sum over the length of the query
        # loss_columnwise = self.ce_loss(scores.T, torch.arange(scores.shape[1], device=scores.device))
        # loss = (loss_rowwise + loss_columnwise) / 2
        return loss_rowwise
    

class ColbertPairwiseCELoss_prune(torch.nn.Module):
    def __init__(self, prune_ratio):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.prune_ratio = prune_ratio


    def prune(self, query_embeddings, doc_embeddings):
        # Compute the ColBERT scores
        scores_prune = torch.einsum("bnd,csd->bcns", doc_embeddings, query_embeddings)
        self_mask = 1 - torch.eye(scores_prune.size(0), device=scores_prune.device).long()
        scores_prune = scores_prune * self_mask.unsqueeze(-1).unsqueeze(-1)
        scores_prune = scores_prune.max(dim=-1).values.max(dim=1).values

        prune_masks = torch.ones((doc_embeddings.size(0), doc_embeddings.size(1)), dtype=torch.long).to(doc_embeddings.device)
        for prune_mask, score_prune in zip(prune_masks, scores_prune):
            zero_num = score_prune.eq(0).sum().item()
            index_rank = score_prune.sort().indices[zero_num:]
            remove_index = index_rank[:int(len(index_rank)*self.prune_ratio)]
            prune_mask[remove_index] = 0
        doc_embeddings = doc_embeddings * prune_masks.unsqueeze(-1)
        return doc_embeddings


    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)

        Positive scores are the diagonal of the scores matrix.
        """
        doc_embeddings = self.prune(query_embeddings, doc_embeddings)
        # Compute the ColBERT scores
        scores = (
            torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
        )  # (batch_size, batch_size)

        # Positive scores are the diagonal of the scores matrix.
        pos_scores = scores.diagonal()  # (batch_size,)

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We exclude the diagonal by setting it to a very low value: since we know the maximum score is 1,
        # we can subtract 1 from the diagonal to exclude it from the maximum operation.
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
        neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores
        # relative to the negative scores.
        # This can be simplified to log-sum-exp of negative scores minus the positive score
        # for numerical stability.
        # torch.vstack((pos_scores, neg_scores)).T.softmax(1)[:, 0].log()*(-1)
        loss = F.softplus(neg_scores - pos_scores).mean()
        print(loss, "prune")
        return loss


class ColbertPairwiseCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)

        Positive scores are the diagonal of the scores matrix.
        """

        # Compute the ColBERT scores
        scores = (
            torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
        )  # (batch_size, batch_size)

        # Positive scores are the diagonal of the scores matrix.
        pos_scores = scores.diagonal()  # (batch_size,)

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We exclude the diagonal by setting it to a very low value: since we know the maximum score is 1,
        # we can subtract 1 from the diagonal to exclude it from the maximum operation.
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
        neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores
        # relative to the negative scores.
        # This can be simplified to log-sum-exp of negative scores minus the positive score
        # for numerical stability.
        # torch.vstack((pos_scores, neg_scores)).T.softmax(1)[:, 0].log()*(-1)
        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss
    

class ColbertPairwiseCELoss_Matryoshka(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)

        Positive scores are the diagonal of the scores matrix.
        """
        assert(math.log2(query_embeddings.size(-1)).is_integer())

        low_exp, high_exp = 5, int(math.log2(query_embeddings.size(-1)))
        loss = 0.0
        for exp in range(low_exp, high_exp+1):
            query_embeddings_sub = F.normalize(query_embeddings[..., :2**exp], p=2, dim=-1)
            doc_embeddings_sub = F.normalize(doc_embeddings[..., :2**exp], p=2, dim=-1)
            # Compute the ColBERT scores
            scores = (
                torch.einsum("bnd,csd->bcns", query_embeddings_sub, doc_embeddings_sub).max(dim=3)[0].sum(dim=2)
            )  # (batch_size, batch_size)
            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)

            # Negative score for a given query is the maximum of the scores against all all other pages.
            # NOTE: We exclude the diagonal by setting it to a very low value: since we know the maximum score is 1,
            # we can subtract 1 from the diagonal to exclude it from the maximum operation.
            neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

            # Compute the loss
            # The loss is computed as the negative log of the softmax of the positive scores
            # relative to the negative scores.
            # This can be simplified to log-sum-exp of negative scores minus the positive score
            # for numerical stability.
            # torch.vstack((pos_scores, neg_scores)).T.softmax(1)[:, 0].log()*(-1)
            loss += F.softplus(neg_scores - pos_scores).mean()
        loss = loss/(high_exp-low_exp+1)
        return loss


class ColbertPairwiseNegativeCELoss(torch.nn.Module):
    def __init__(self, in_batch_term=False):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.in_batch_term = in_batch_term

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        neg_doc_embeddings: (batch_size, num_neg_doc_tokens, dim)
        """

        # Compute the ColBERT scores
        pos_scores = torch.einsum("bnd,bsd->bns", query_embeddings, doc_embeddings).max(dim=2)[0].sum(dim=1)
        neg_scores = torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings).max(dim=2)[0].sum(dim=1)

        loss = F.softplus(neg_scores - pos_scores).mean()

        if self.in_batch_term:
            scores = (
                torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
            )  # (batch_size, batch_size)

            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)
            neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

            loss += F.softplus(neg_scores - pos_scores).mean()

        return loss / 2
