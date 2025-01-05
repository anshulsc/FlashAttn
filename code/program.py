import triton 
import torch

import triton.language as tl


def test_op( BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype = torch.float16 ):

    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=torch.device('cuda')
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(True)
    )

    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=torch.device('cuda')
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(True)
    )

    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=torch.device('cuda')
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(True)
    )

    softmax_scale = 1.0 / (HEAD_DIM ** 0.5)