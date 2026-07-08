# Understanding Qwen3.5-2B with SAEs

Qwen provides a sparse autoencoder for Qwen3.5-2b called SAE-Res-Qwen3.5-2B-Base-W32K-L0_100.
This SAE is trained on the residual stream (the outputs being added from one hidden layer to the next) of Qwen3.5-2B. It has 32K hidden features with the top 100 kept active (non-zero) per token while every everything else is zeroes (top k architecture).

## Implications

This is cool because it means if you have access to the weights you can potentially avoid SFT for tuning response tone in the model. You could use a SAE to find features that correspond to "friendly" for example, and emphasise them at inference time.
