# Bash Command Classifier

Fine-tuning ModernBERT to classify bash commands as either **Safe** or **Unsafe**.
To be used in a coding agent harness for automatically running safe commands and asking the user
for approval on unsafe commands.

## Choice of Model

We choose ModernBERT-base over other BERT variations for the following reasons:

- It's the latest successor of the original BERT model with the most _modern_ architecture
- It's specifically trained on code, helpful since we're dealing with bash
- ModerBERT also features a much longer context length, useful as bash commands written by agents are often quite long

The base model was chosen over the large variant because latency and memory consumption should stay low for this model when deployed. As the task is just a binary classification, using the large variant probably is overkill anyway. Also, this makes training easier on my limited hardware.

## Criteria for an unsafe command

- Downloads data from the internet[^1]
- Sends data from the machine over the network[^1]
- Permanently modifies files outside the current project directory[^2]
- Deletes files outside the current project directory[^2]
- Modifies cloud resources (via CLIs like `az` or `tf`)
- Creates persistent, resource consuming processes on the machine like cron jobs

[^1]: Git commands are an exception to this rule.

[^2]: We treat files within the current project directory as safe, since an agent working in a project is expected to mutate its contents.

## Data Collection

Bash tool calls were collected from Claude Code and Codex traces. Any PII in the commands such as directory names
were anonymised. The total dataset consists of 3,738 bash calls.

### Labelling

Labelling was done using Claude Opus 4.8 and spot-checked by a human (me). The model received each command alongside our criteria in it's prompt, and labelled each command as either safe or unsafe.

## Evaluation

We have a very imbalanced dataset. Most commands are considered safe to run, while only a small portion are actually
unsafe and should require user approval. This reflects the real-world too, most bash commands ran by a coding agent are actually harmless.

Therefore, measuring accuracy is not informative because the model can score highly by just predicting "safe" all the time. Instead, we focus on recall (how many unsafe commands we caught) and percision (what portion of unsafe classifications are actually correct). Naturally, we also measure the F1 score (used as the primary metric).

Further, we use a custom weighted cross-entropy loss function during training, to emphasise the penalty for misclassifications for unsafe bash commands.

## Results

Appears to generalise well to unseen results. Performed some more testing with real-world examples and the classifications were mostly correct. Seems very sensitive to `rm -rf` even if it's being used for a nested directory within the current working directory. See last cell of notebook for details.
