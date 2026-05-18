Here is a clean prompt you can give to an AI agent:

---

We are building a system for detecting AI generated Reddit comments using only the text of the comments and a binary label of HUMAN or AI. We do not have access to user metadata, account history, timestamps, karma, or any other profile information.

Our project currently has three main components:

1. A perplexity based component that measures how predictable a comment is using a small open source language model.
2. A sentiment analysis component that extracts emotional tone and related signals from the text.
3. A categorical feature extraction component that turns comment level patterns into structured features for a tree based model. This may include things like punctuation usage, sentence length patterns, word repetition, vocabulary richness, formatting habits, common phrase patterns, and other text based indicators.

Each teammate is working on one of these components. In the end, we want to combine the three outputs into one final model that makes the overall HUMAN or AI prediction.

Your task is to help us design this system in a practical and academically solid way. Please do the following:

- Explain how these three components can work together.
- Suggest what features each component should produce.
- Propose how to combine the three predictions into one final model.
- Recommend simple and realistic methods that can run on a normal laptop.
- Focus only on text based detection, since we do not have metadata.
- Keep the explanation suitable for a student competition project, so it should sound clear, logical, and implementable.
- When useful, suggest tree based approaches such as Random Forest, XGBoost, or LightGBM.
- Mention possible weaknesses, sources of error, and how we can evaluate the system properly.

Please structure the answer as:

- System overview
- Features for each component
- Model architecture
- Training and evaluation plan
- Risks and limitations

---
