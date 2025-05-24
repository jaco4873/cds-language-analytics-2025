"""N-gram Language Model Implementation

A statistical language model implementation supporting variable n-gram sizes with
smoothing and backoff techniques.

Key Features:
    - Variable n-gram order (1 to n)
    - Laplace (add-1) smoothing
    - Stupid backoff (α=0.4)
    - Multiple text generation sampling methods (top-k, nucleus, temperature)

Implementation Details:
    - Uses NLTK for tokenization
    - Maintains frequency counts and conditional distributions
    - Supports model persistence via JSON
    - Implements hierarchical backoff models

Usage Example:
    model = NgramModel("trigram-model", n_gram_size=3)
    model.train("training_data_path")
    generated_text = model.generate(seed="The quick brown", tokens=50)
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from nltk import word_tokenize, ngrams
from tqdm import tqdm

from ..config.settings import settings
from ..utils.logger import logger


class NgramModel:
    def __init__(
        self,
        name: str,
        n_gram_size: int = 2,
        *,
        use_smoothing: bool = True,
        use_stupid_backoff: bool = True,
    ):
        """Initialize an n-gram language model.

        Args:
            name: Identifier for the model, used for saving/loading
            n_gram_size: Size of n-grams (e.g., 2 for bigrams, 3 for trigrams)
            use_smoothing: Whether to use Laplace (add-1) smoothing to handle rare/unseen events
            use_stupid_backoff: Whether to use stupid backoff for unseen histories
                              (backs off to lower-order n-grams with penalty α)
        """
        self.name = name
        self.n_gram_size = n_gram_size
        self.use_smoothing = use_smoothing
        self.use_stupid_backoff = use_stupid_backoff
        self.ngram_counter = Counter()  # Raw n-gram counts
        self.model = defaultdict(Counter)  # Conditional distributions
        self.vocab = set()
        self.backoff_models = []

    def train(self, folder_path: str):
        """Train the n-gram model on text files.

        Args:
            folder_path: Path to directory containing .txt files

        The training process:
        1. Reads all .txt files in the directory
        2. Builds vocabulary and counts n-gram frequencies
        3. If stupid backoff is enabled, creates and trains a chain of
           lower-order models (n-1, n-2, ..., 1) for backoff
        """
        folder = Path(folder_path)
        files = list(folder.glob("*.txt"))

        # First pass: collect counts
        logger.info(f"Training {self.n_gram_size}-gram model...")
        for file in tqdm(files, desc="Counting n-grams"):
            with open(file, "r") as f:
                text = f.read()
            tokens = word_tokenize(text)
            self.vocab.update(tokens)
            text_ngrams = ngrams(tokens, self.n_gram_size)
            self.ngram_counter.update(text_ngrams)

        # Second pass: build distributions
        logger.info("Building probability distributions...")
        for ngram, count in self.ngram_counter.items():
            history, continuation = ngram[:-1], ngram[-1]
            self.model[history][continuation] += count

        # If using backoff, train smaller n-gram models
        if self.use_stupid_backoff and self.n_gram_size > 1:
            logger.info("Training backoff models...")
            for order in range(self.n_gram_size - 1, 0, -1):
                backoff = NgramModel(
                    f"{self.name}-{order}gram",
                    n_gram_size=order,
                    use_smoothing=self.use_smoothing,
                    use_stupid_backoff=False,  # Prevent infinite recursion (no backoff for unigrams)
                )
                backoff.train(folder_path)
                self.backoff_models.append(backoff)

    def conditional_probability_distribution(
        self, history: tuple[str, ...]
    ) -> dict[str, float]:
        """Get p(w|history) with optional smoothing/backoff"""
        if history not in self.model:
            if self.use_stupid_backoff and self.backoff_models:
                logger.debug(
                    f"Backoff: Using {len(history) - 1}-gram model for unseen history {history}"
                )
                shorter_hist = history[1:] if len(history) > 0 else tuple()
                backoff_dist = self.backoff_models[
                    0
                ].conditional_probability_distribution(shorter_hist)
                return {w: p * settings.BACKOFF_ALPHA for w, p in backoff_dist.items()}
            # Fallback to uniform distribution
            logger.debug(
                f"Fallback: Using uniform distribution for unseen history {history}"
            )
            prob = 1.0 / len(self.vocab)
            return {word: prob for word in self.vocab}

        counts = self.model[history]
        total = sum(counts.values())

        if self.use_smoothing:
            # Laplace (add-1) smoothing
            vocab_size = len(self.vocab)
            dist = {
                word: (counts.get(word, 0) + 1) / (total + vocab_size)
                for word in self.vocab
            }
            # Debug log the top 3 predictions
            top3 = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3])
            logger.debug(f"Smoothed predictions for {history}: {top3}")
            return dist

        # Standard MLE
        dist = {word: count / total for word, count in counts.items()}
        top3 = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3])
        logger.debug(f"MLE predictions for {history}: {top3}")
        return dist

    def generate(
        self,
        seed: tuple[str, ...] | None = None,
        tokens: int = 25,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> str:
        """Generate text with optional sampling methods

        Args:
            seed: Starting sequence of tokens
            tokens: Number of tokens to generate
            top_k: If set, only sample from the k most likely tokens
            top_p: If set, sample from the smallest set of tokens whose cumulative probability exceeds p
            temperature: Softmax temperature (higher = more random, lower = more deterministic)

        Raises:
            ValueError: If both top_k and top_p are specified, or if temperature <= 0
        """
        if len(self.vocab) == 0:
            raise ValueError("Empty vocabulary - model needs training first")

        if top_k is not None and top_p is not None:
            raise ValueError(
                "Cannot use both top-k and nucleus sampling. Please specify only one."
            )

        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        if seed is None:
            # Random start from existing histories
            seed = random.choice(list(self.model.keys()))

        generated = list(seed)

        for _ in range(tokens):
            history = tuple(generated[-(self.n_gram_size - 1) :])
            dist = self.conditional_probability_distribution(history)

            if not dist:
                next_token = random.choice(list(self.vocab))
            else:
                # Handle sampling methods
                sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)

                if top_k:
                    sorted_items = sorted_items[:top_k]

                if top_p:
                    # Nucleus sampling
                    cumsum = 0.0
                    for i, (_, prob) in enumerate(sorted_items):
                        cumsum += prob
                        if cumsum > top_p:
                            sorted_items = sorted_items[: i + 1]
                            break

                words, probs = zip(*sorted_items)

                if temperature != 1.0:
                    probs = [p ** (1.0 / temperature) for p in probs]

                # Normalize and sample
                total = sum(probs)
                probs = [p / total for p in probs]
                next_token = random.choices(words, weights=probs)[0]

            generated.append(next_token)

        return " ".join(generated)

    def save(self, models_path: str = str(settings.MODELS_DIR)):
        """Save model state to JSON"""
        model_path = Path(models_path) / f"{self.name}.json"

        # Handle n-gram counts differently for unigrams vs n-grams
        if self.n_gram_size == 1:
            # For unigrams, values are already integers
            counts_data = {str(k): v for k, v in self.ngram_counter.items()}
        else:
            # For n-grams, convert tuples to strings
            counts_data = {str(k): dict(v) for k, v in self.model.items()}

        model_data = {
            "name": self.name,
            "n_gram_size": self.n_gram_size,
            "use_smoothing": self.use_smoothing,
            "use_stupid_backoff": self.use_stupid_backoff,
            "ngram_counts": counts_data,
            "vocab": list(self.vocab),
        }

        with open(model_path, "w") as f:
            json.dump(model_data, f)

        # Save backoff models if they exist
        for model in self.backoff_models:
            model.save(models_path)

    @classmethod
    def load(cls, model_name: str, models_path: str = str(settings.MODELS_DIR)):
        """Load model from JSON"""
        model_path = Path(models_path) / f"{model_name}.json"

        with open(model_path, "r") as f:
            model_data = json.load(f)

        model = cls(
            model_data["name"],
            model_data["n_gram_size"],
            use_smoothing=model_data.get("use_smoothing", False),
            use_stupid_backoff=model_data.get("use_stupid_backoff", False),
        )

        # Restore core data
        model.vocab = set(model_data["vocab"])

        # Handle counts differently for unigrams vs n-grams
        if model.n_gram_size == 1:
            # For unigrams, values are integers
            model.ngram_counter.update(
                {
                    tuple([eval(k)]): v  # Wrap single token in tuple
                    for k, v in model_data["ngram_counts"].items()
                }
            )
        else:
            # For n-grams, need to rebuild both counter and model
            for history_str, continuations in model_data["ngram_counts"].items():
                history = tuple(eval(history_str))
                model.model[history].update(continuations)
                # Rebuild counter from model
                for word, count in continuations.items():
                    model.ngram_counter[history + (word,)] = count

        # Load backoff models if needed
        if model.use_stupid_backoff:
            for order in range(model.n_gram_size - 1, 0, -1):
                try:
                    backoff = cls.load(f"{model_name}-{order}gram", models_path)
                    model.backoff_models.append(backoff)
                except FileNotFoundError:
                    logger.warning(f"Could not find {order}-gram backoff model")
                    break

        return model
