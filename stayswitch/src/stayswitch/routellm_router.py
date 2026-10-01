"""RouteLLM's BERT router applied per step (baseline B3).

RouteLLM (Ong et al., ICLR 2025) trains single-turn routers on Chatbot Arena
preferences; its BERT router scores a prompt with a 3-way classifier (strong
wins / tie / weak wins) and routes to the strong model when
``1 - P(tie or weak wins)`` exceeds a threshold. This reproduces that scoring
with the released checkpoint and applies it to every agent call, which is how a
single-turn router is typically dropped into an agent.

The prompt it scores is the text the model must respond to next: the task
statement on the first call, the latest terminal output afterwards.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from stayswitch.policy import Decision
from stayswitch.session import SessionState
from stayswitch.signals import _text


class BertScorer:
    def __init__(self, checkpoint: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
        self.model.eval()

    def strong_win_rate(self, prompt: str) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits.numpy()[0]
        scores = np.exp(logits - np.max(logits))
        scores /= scores.sum()
        # Same reduction as routellm.routers.routers.BERTRouter.calculate_strong_win_rate.
        return float(1 - np.sum(scores[-2:]))


class RouteLLMPolicy:
    def __init__(self, weak: str, strong: str, threshold: float, checkpoint: str) -> None:
        self.weak, self.strong, self.threshold = weak, strong, threshold
        self.scorer = BertScorer(checkpoint)

    def decide(self, state: SessionState, messages: list[dict[str, Any]]) -> Decision:
        user_turns = [m for m in messages if m.get("role") == "user"]
        prompt = _text(user_turns[-1].get("content")) if user_turns else ""
        # The first turn is instructions + task + initial screen; keep the task-bearing tail.
        prompt = prompt[-3000:]
        p = self.scorer.strong_win_rate(prompt)
        return Decision(self.strong if p >= self.threshold else self.weak, note=f"p={p:.3f}")
