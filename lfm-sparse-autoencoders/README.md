# LFM2.5-230M Sparse Autoencoders

LFM2.5-230M is an interesting model. It's not a transformer, it's something hybrid, made up
of 8 double-gated Linear Input-Varying (LIV) convolution blocks and 6 Grouped Query Attention (GQA) blocks. The LIV conv blocks act as an alternative attention mechanism while the GQA blocks helps with associative recall.

It turns out (unsurprisingly), you can use Sparse Autoencoders to do weight surgery on this type of model architecture too. At the end of the day, it too has a residual stream.

Thus, I trained BatchTopK SAEs on the residual stream of LFM2.5-230M.

## Training

I ran training on two Kaggle T4 GPUs. Since each model comfortably fits on a
single GPU, I was able to train two separate SAEs, one at layer 6 and another at layer 10.

One issue I ran into was LFM2.5-230M comes in bf16 percision (like most modern models) but T4 is a very old architecture and doesn't support bf16. So, I had to cast to fp16 instead. This effectively means the SAE was trained on activations produced at a higher precision. The only concern was whether this would cause overflow (it has 5bit exponent which can't handle large values that bf16 could with a 8bit exponent). This can be easily validated though by checking all the hidden states (what we care about) are finite under fp16 and they were.

## Eval and Findings

| Metric                | Layer 6 | Layer 10 |
| --------------------- | ------: | -------: |
| L0                    |   65.17 |    69.09 |
| Explained variance    |  0.8511 |   0.8517 |
| Dead feature fraction |  0.0012 |   0.0035 |

The explained variance is ~85% at both layer 6 and layer 10 SAEs. L0 is near K (64) and the near-zero dead feature fraction shows we have good utilization.

From the eval logs, it appears layer 6 feature activations are mostly single token definitions like `fine`, `here`, `such`, while layer 10 feature activations are more _conceptual_, like `specifying`, `livestock`, `gameplay`. This makes sense as later hidden layers are persumed to encode more abstract meanings into latents.

The BOS position (residual vector at the beginning-of-sequence token) is a huge outlier. The results showed a ~50x reconstruction error of everything else. Models tend to use that slot as a scratchpad so we exclude special token positions from training and eval.

The residual stream is small-normed (~0.8-2.2 across all 14 layers) until the final RMSNorm scales it by ~40x.

## Baking in Math Refusal

As a test, I use the SAE to find what features fire in prompts related to solving maths, and induce a refusal in the model to solve maths. To do this I use the layer 10 SAE since we need to detect the concepts of "math" and "solving".
