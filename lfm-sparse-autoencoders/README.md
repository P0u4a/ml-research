# LFM2.5-230M Sparse Autoencoders

LFM2.5-230M is an interesting model. It's not a transformer, it's something hybrid, made up
of 8 double-gated Linear Input-Varying (LIV) convolution blocks and 6 Grouped Query Attention (GQA) blocks. The LIV conv blocks act as an alternative attention mechanism while the GQA blocks helps with associative recall.

It turns out (unsurprisingly), you can use Sparse Autoencoders to do weight surgery on this type of model architecture too. At the end of the day, it too has a residual stream. 

Thus, I trained BatchTopK SAEs on the residual stream of LFM2.5-230M.

## Training

I ran training on two Kaggle T4 GPUs. Since each model comfortably fits on a
single GPU, I was able to train two separate SAEs, one at layer 6 and another at layer 10.

One issue I ran into was LFM2.5-230M comes in bf16 percision (like most modern models) but T4 is a very old architecture and doesn't support bf16. So, I had to cast to fp16 instead. This effectively means the SAE was trained on activations produced at a higher precision. The only concern was whether this would cause overflow (it has 5bit exponent which can't handle large values that bf16 could with a 8bit exponent). This can be easily validated though by checking all the hidden states (what we care about) are finite under fp16 and they were.

## Eval

The SAE trained at layer 10 exhibited better performance than the one trained at layer 6.

## Baking in Math Refusal

As a test, I use the SAE to find what features fire in prompts related to solving maths, and induce a refusal in the model to solve maths.
