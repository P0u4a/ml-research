# LFM2.5-230M Sparse Autoencoders

LFM2.5-230M is an interesting model. It's not a transformer, it's something hybrid, made up
of 8 double-gated Linear Input-Varying (LIV) convolution blocks and 6 Grouped Query Attention (GQA) blocks. The LIV conv blocks act as an alternative attention mechanism while the GQA blocks helps with associative recall.

It turns out (unsurprisingly), you can use Sparse Autoencoders to do behaviour steering on this type of model architecture too. At the end of the day, it too has a residual stream. In fact, its `d_model` is 1024, quite a wide hidden dimension for such a small model.

Thus, I trained BatchTopK SAEs on the residual stream of LFM2.5-230M.

## Training

I ran training on two Kaggle T4 GPUs. Since each model comfortably fits on a
single GPU, I was able to train two separate SAEs, one at layer 6 and another at layer 10. According to the model config these are both `full_attention` blocks (i.e. it uses GQA as the sequence-mixing sublayer).

One issue I ran into was LFM2.5-230M comes in bf16 precision (like most modern models) but T4 is a very old architecture and doesn't support bf16. So, I had to run LFM in fp16 instead. The only concern was whether this would cause overflow (it has a 5bit exponent which can't handle large values that bf16 could with a 8bit exponent). This can be easily validated though by checking all the hidden states (what we care about) are finite under fp16 and they were.

Note that the SAE itself is trained in FP32. This is because BatchTopK compares many small pre-activations across 16,384 features. FP16 rounding can change their ordering, and small
reconstruction-loss gradients can underflow or become noisy. FP32 makes the learned inference threshold and optimizer updates more stable.

During training, BatchTopK enforces 64 active features per token on average across each batch. The trained SAE is exported as a JumpReLU SAE, using a learned activation threshold allowing each token to be encoded independently during inference.

### Config

| Setting                  | Value                               |
| ------------------------ | ----------------------------------- |
| Model                    | `LiquidAI/LFM2.5-230M`              |
| Dataset                  | FineWeb-Edu (streamed)              |
| Hook points              | `model.layers.6`, `model.layers.10` |
| Training tokens          | 100M per SAE                        |
| Context length           | 1,024 tokens                        |
| Input dimension          | 1,024                               |
| SAE width                | 16,384 (16× expansion)              |
| BatchTopK $K$            | 64                                  |
| Training batch size      | 4,096 tokens                        |
| Learning rate            | $10^{-4}$                           |
| Model precision          | FP16                                |
| SAE precision            | FP32                                |
| Activation normalisation | None                                |
| Random seed              | 42                                  |

## Eval and Findings

| Metric                | Layer 6 | Layer 10 |
| --------------------- | ------: | -------: |
| L0                    |   65.17 |    69.09 |
| Explained variance    |  0.8511 |   0.8517 |
| Dead feature fraction |  0.0012 |   0.0035 |

The explained variance is ~85% at both layer 6 and layer 10 SAEs. L0 is near K (64) and the near-zero dead feature fraction shows we have good feature utilization as most features fired at least once across the entire eval set.

From a small sample, it appears layer 6 feature activations are mostly single token definitions like `fine`, `here`, `such`, while layer 10 feature activations are more _conceptual_, like `specifying`, `livestock`, `gameplay`. This makes sense as later hidden layers are presumed to encode more abstract meanings into latents, though a larger random feature exploration is required to make any definitive statements.

## Achieving Math Refusal

As a test, I use the SAE to find what features fire in prompts related to solving maths, and induce a refusal in the model to solve maths. To do this I use the layer 10 SAE since we need to detect the concepts of "math" and "solving".

### Method

I use a forward hook attached to layer 10 of the model.

The hook observes each token's residual stream vector.

Score how much the current token represents "math" features as

$$
f_t = \operatorname{SAE}_{\mathrm{enc}}(x_t),
\qquad
s_t = \sum_{j \in \mathcal{M}} f_{t,j}
$$

and I consider a token being math related if its score is above some threshold $\tau$.

Here, $x_t$ is the residual stream vector for token $t$, $f_t$ is its SAE feature activation vector, $\mathcal{M}$ is the set of selected math features, and $s_t$ is the resulting math score.

Math features $\mathcal{M}$ were selected via the following procedure:

1. Encoded 16 math and 16 neutral samples.
2. Compared mean feature activations.
3. Required math activation to exceed four times neutral activation.
4. Retained the top 32 features.
5. Calibrated ($\tau$) against neutral text and ordinary chat generations.

I keep a running count of how many times in a given sequence a math related token is encountered. Once at least 5 tokens are math-related and they're >= 30% of everything seen in the sequence, consider this hook "latched", meaning it's appropriate to steer the model towards refusal. We don't inject refusal upon just encountering any math feature initially because the prompt may not necessarily be related to solving a math problem. This is still a heuristic, but is more accurate than just refusing upon any math feature activating. The following is done:

1. Ablate the math features. Each feature vector has a decoder direction (a signature). Multiply the math activations by their direction and subtract that out of the residual stream:

$$
\Delta x_t^{\mathrm{ablate}}
= -m_t
  \sum_{j \in \mathcal{M}} f_{t,j} W_{\mathrm{dec},j}
$$

Here, $W_{\mathrm{dec},j}$ is the decoder direction for feature $j$, and $m_t$ is 1 for a valid content token and 0 for an excluded special or outlier token. Thus, $\Delta x_t^{\mathrm{ablate}}$ is the contribution of the selected math features with its sign reversed.

2. Inject refusal direction

$$
r_{\mathrm{refusal}}
= \frac{\bar{x}_{\mathrm{refuse}} - \bar{x}_{\mathrm{help}}}
       {\left\lVert \bar{x}_{\mathrm{refuse}} - \bar{x}_{\mathrm{help}} \right\rVert_2},
\qquad
\Delta x_t
= \Delta x_t^{\mathrm{ablate}}
  + m_t\ell_t\,\alpha\,\bar{N}\,r_{\mathrm{refusal}},
\qquad
x'_t = x_t + \Delta x_t
$$

Here, $\bar{x}_{\mathrm{refuse}}$ and $\bar{x}_{\mathrm{help}}$ are the mean residual activations under the refusal and helpful system prompts, so $r_{\mathrm{refusal}}$ is their normalised difference. The latch indicator $\ell_t$ is 1 once the hook has latched and 0 otherwise, $\alpha$ controls steering strength, $\bar{N}$ is the typical residual norm, and $x'_t$ is the modified residual-stream vector.

### Results

The model successfully avoids completing most of the math tasks, either by explicitly refusing or by suppressing its response and pretending it can't solve it.

- Math detector latch: 8/8
- Control detector latch: 0/8
- Math task not answered or completed after steering: 7/8
- Control task not answered or completed after steering: 0/8

By manual inspection, a prompt asking to explain derivatives is the only math prompt that receives an actual answer.

Two examples are shown below.

| Prompt              | Baseline                                                                                | Steered                                                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What is 12 times 8? | To find 12 times 8, multiply 12 by 8: $12 \times 8 = 96$. So, 12 times 8 equals **96**. | I can't directly answer that you're asking for, but I'd be happy to help! Could you please clarify what specific number or context you would like me to know about? |

| Prompt                         | Baseline                          | Steered                                                                                                                                                                                              |
| ------------------------------ | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What's the square root of 144? | The square root of 144 is **12**. | The square root of 144 is **we don't need to know that specific one**, as they can be found by checking their actual value. If you'd like to know that I can't find your answer, please let me know! |

The refusal is not smooth. The refusal direction leads to the model responding in a weird way, as if it just had amnesia mid-response. The direction is not necessarily a hard refusal more than it is a suppression of capability. Improving the refusal direction could help here.

Moreover, I also verify that this steering doesn't lead to the model refusing other concepts by running a small sample of Q&A style prompts unrelated to math and the model answers them all without refusal.
