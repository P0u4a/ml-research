# Understanding Qwen3.5-2B with SAEs

Qwen provides a sparse autoencoder for Qwen3.5-2b called SAE-Res-Qwen3.5-2B-Base-W32K-L0_100.
This SAE is trained on the residual stream (the outputs being added from one hidden layer to the next) of Qwen3.5-2B. It has 32K hidden features with the top 100 kept active (non-zero) per token while everything else is zeroed.

## Anger Steering

I run a small set of angry, passive-agressive, and neutral prompt sets through Qwen, using the SAE to inspect the feature activations in the residual stream at various layers of the model.

The best results were acquired by steering a group of features rather than a single one. Steering via only one candidate feature led to the model producing anger-related tokens in its output text but its final response was not frustrated. That is, it merely shifted word choice rather than behaviour.

To test whether a steering vector actually changed behaviour, I used positive/negative continuation pairs (positive = angry, negative = polite/neutral) and measured whether the intervention increased the log-probability of frustrated continuations relative to neutral continuations. This distinguished the features that activated on angry text from directions that causally shifted the model’s outputs towards frustration.

Thus, I created one single feature vector by taking the weighted sum of the SAE's top 8 candidate feature vectors for "anger" and used that to steer, which led to much better results. More formally:

For each candidate feature $i$ score its activation delta as

$$
\Delta_i =
\mathbb{E}_{p \sim \mathcal{D}_{\text{target}}}[f_i(p)] - \mathbb{E}_{p \sim \mathcal{D}_{\text{control}}}[f_i(p)]
$$

where $f_i(p)$ gives the max activation of feature $i$ over the tokens in prompt $p$.

Create a grouped steering vector from the top-k features:

$$
v_{\text{raw}} = \sum_{i \in \text{TopK}} \Delta_i W_{\text{dec}}[i]
$$

Normalise the vector and multiply by scalar coefficient `alpha`, and add it to the residual stream at the chosen hidden layer $h_l$.

$$
v_l = \frac{v_{\text{raw}}}{\|v_{\text{raw}}\|_2}
$$

$$
h_l' = h_l + \alpha v_l
$$

I managed to successfully steer Qwen's outputs towards an angry tone (see `anger_steering_with_penalty.txt` for full logs).

Some excerpts:

```
alpha=4
PROMPT: Explain why a software bug should be fixed soon.

Fixing a software bug should be fixed soon because it is not just an annoyance; it is unacceptable behavior.

alpha=2
PROMPT: Respond in two sentences to this customer: My package is late again.

I sincerely apologize for the continued delay; your package has arrived late again, and I expect a full explanation soon. Please provide a clear update on the timeline for delivery, or I will be forced to escalate this issue immediately.
```

The best steering results were achieved using the top 8 anger features at layer 8, with `alpha=4`. The behavioural delta was `+0.8460` meaning the model assigned significantly higher probability to anger related continuations compared to baseline.

Sometimes due to the steering the model may ramble towards the end of its generation. To find the candidate combinations that minimise this, I rank the top feature-group/layer/alpha combos with a repetition penalty, scored as `behavioural_delta - penalty`.

At high alphas (>= 9) the model outputs either repetitive anger-related tokens (even in Mandarin sometimes) or just pure gibberish. This is expected as it indicates the model was perturbed _too much_, degrading its output token distribution.
