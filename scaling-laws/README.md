# Scaling Laws

<img width="548" height="335" alt="image" src="https://github.com/user-attachments/assets/33478fc4-b72c-43cd-aa19-a1c593bdb944" />

Reasoning capability generally increases with parameter count, but not strictly. 

- Exhibit A
  - Qwen3.6-27B outperforms several 100B+ parameter models (e.g. Deepseek V3, Deepseek R1, GPT-OSS-120B). 
- Exhibit B
  - Gemma-3-12B is outperformed by it's smaller successor Gemma-4-E2B.

So what's the secret sauce? 

According to Chinchilla scale, you need about ~20 tokens of training data per parameter in the model (if the goal is to optimise for training-compute budget).
But, you can train a much smaller model on a much larger number of tokens and outperform larger chinchilla-scaled models by paying the cost of training compute.

