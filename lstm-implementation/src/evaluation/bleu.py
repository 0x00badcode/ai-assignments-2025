"""
BLEU score implementation for evaluating neural machine translation quality.
Based on the original BLEU paper: "BLEU: a Method for Automatic Evaluation of Machine Translation"
"""
import numpy as np
from collections import Counter, defaultdict
import math

def tokenize(text):
    """Simple tokenization (split by whitespace)."""
    if isinstance(text, str):
        return text.strip().split()
    return text

def compute_ngrams(tokens, n):
    """Compute n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def count_ngrams(tokens, max_n=4):
    """Count n-grams up to max_n."""
    ngram_counts = defaultdict(Counter)
    for n in range(1, max_n + 1):
        ngrams = compute_ngrams(tokens, n)
        ngram_counts[n] = Counter(ngrams)
    return ngram_counts

def compute_precision(candidate_ngrams, reference_ngrams, n):
    """Compute precision for n-grams."""
    if len(candidate_ngrams[n]) == 0:
        return 0.0
    
    clipped_count = 0
    total_count = 0
    
    for ngram, count in candidate_ngrams[n].items():
        # Get maximum count from all references
        max_ref_count = max(ref_ngrams[n].get(ngram, 0) for ref_ngrams in reference_ngrams)
        clipped_count += min(count, max_ref_count)
        total_count += count
    
    return clipped_count / total_count if total_count > 0 else 0.0

def brevity_penalty(candidate_length, reference_length):
    """Compute brevity penalty."""
    if candidate_length > reference_length:
        return 1.0
    elif candidate_length == 0:
        return 0.0
    else:
        return math.exp(1 - reference_length / candidate_length)

def sentence_bleu(candidate, references, weights=(0.25, 0.25, 0.25, 0.25), smoothing=False):
    """
    Compute BLEU score for a single sentence.
    
    Args:
        candidate: Candidate translation (string or list of tokens)
        references: List of reference translations (list of strings or list of token lists)
        weights: Weights for n-gram precisions (default: uniform weights for 1-4 grams)
        smoothing: Whether to apply smoothing for zero counts
    
    Returns:
        BLEU score
    """
    # Tokenize inputs
    candidate_tokens = tokenize(candidate)
    reference_tokens_list = [tokenize(ref) for ref in references]
    
    # Compute n-gram counts
    candidate_ngrams = count_ngrams(candidate_tokens, len(weights))
    reference_ngrams_list = [count_ngrams(ref_tokens, len(weights)) 
                           for ref_tokens in reference_tokens_list]
    
    # Compute precisions
    precisions = []
    for n in range(1, len(weights) + 1):
        precision = compute_precision(candidate_ngrams, reference_ngrams_list, n)
        
        # Apply smoothing if enabled and precision is 0
        if smoothing and precision == 0:
            precision = 1.0 / (2 ** n)  # Simple smoothing
        
        precisions.append(precision)
    
    # If any precision is 0, BLEU score is 0
    if any(p == 0 for p in precisions):
        return 0.0
    
    # Compute geometric mean of precisions
    log_precisions = [math.log(p) for p in precisions]
    geometric_mean = math.exp(sum(w * log_p for w, log_p in zip(weights, log_precisions)))
    
    # Compute brevity penalty
    candidate_length = len(candidate_tokens)
    # Choose reference length closest to candidate length
    reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]
    closest_ref_length = min(reference_lengths, key=lambda x: abs(x - candidate_length))
    
    bp = brevity_penalty(candidate_length, closest_ref_length)
    
    return bp * geometric_mean

def corpus_bleu(candidates, references_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing=False):
    """
    Compute corpus-level BLEU score.
    
    Args:
        candidates: List of candidate translations
        references_list: List of lists of reference translations
        weights: Weights for n-gram precisions
        smoothing: Whether to apply smoothing
    
    Returns:
        Corpus BLEU score
    """
    # Aggregate counts across all sentences
    total_candidate_ngrams = [Counter() for _ in range(len(weights))]
    total_reference_ngrams = [Counter() for _ in range(len(weights))]
    total_candidate_length = 0
    total_reference_length = 0
    
    for candidate, references in zip(candidates, references_list):
        # Tokenize
        candidate_tokens = tokenize(candidate)
        reference_tokens_list = [tokenize(ref) for ref in references]
        
        # Count n-grams
        candidate_ngrams = count_ngrams(candidate_tokens, len(weights))
        reference_ngrams_list = [count_ngrams(ref_tokens, len(weights)) 
                               for ref_tokens in reference_tokens_list]
        
        # Aggregate candidate n-grams
        for n in range(len(weights)):
            total_candidate_ngrams[n].update(candidate_ngrams[n + 1])
        
        # Aggregate reference n-grams (take maximum count across references)
        for n in range(len(weights)):
            ref_ngram_counts = Counter()
            for ref_ngrams in reference_ngrams_list:
                for ngram, count in ref_ngrams[n + 1].items():
                    ref_ngram_counts[ngram] = max(ref_ngram_counts[ngram], count)
            total_reference_ngrams[n].update(ref_ngram_counts)
        
        # Aggregate lengths
        total_candidate_length += len(candidate_tokens)
        reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]
        closest_ref_length = min(reference_lengths, key=lambda x: abs(x - len(candidate_tokens)))
        total_reference_length += closest_ref_length
    
    # Compute precisions
    precisions = []
    for n in range(len(weights)):
        candidate_count = sum(total_candidate_ngrams[n].values())
        if candidate_count == 0:
            precision = 0.0
        else:
            clipped_count = 0
            for ngram, count in total_candidate_ngrams[n].items():
                clipped_count += min(count, total_reference_ngrams[n].get(ngram, 0))
            precision = clipped_count / candidate_count
        
        # Apply smoothing if enabled
        if smoothing and precision == 0:
            precision = 1.0 / (2 ** (n + 1))
        
        precisions.append(precision)
    
    # If any precision is 0, BLEU score is 0
    if any(p == 0 for p in precisions):
        return 0.0
    
    # Compute geometric mean
    log_precisions = [math.log(p) for p in precisions]
    geometric_mean = math.exp(sum(w * log_p for w, log_p in zip(weights, log_precisions)))
    
    # Compute brevity penalty
    bp = brevity_penalty(total_candidate_length, total_reference_length)
    
    return bp * geometric_mean

class BLEUScore:
    """BLEU score evaluator."""
    
    def __init__(self, max_n=4, weights=None, smoothing=False):
        """
        Initialize BLEU evaluator.
        
        Args:
            max_n: Maximum n-gram order
            weights: Weights for n-gram precisions (default: uniform)
            smoothing: Whether to apply smoothing
        """
        self.max_n = max_n
        self.weights = weights or tuple(1.0 / max_n for _ in range(max_n))
        self.smoothing = smoothing
    
    def sentence_score(self, candidate, references):
        """Compute sentence-level BLEU score."""
        return sentence_bleu(candidate, references, self.weights, self.smoothing)
    
    def corpus_score(self, candidates, references_list):
        """Compute corpus-level BLEU score."""
        return corpus_bleu(candidates, references_list, self.weights, self.smoothing)
    
    def evaluate_model(self, model, test_data, tokenizer=None):
        """
        Evaluate a model using BLEU score.
        
        Args:
            model: Model with translate method
            test_data: List of (source, target) pairs
            tokenizer: Optional tokenizer for detokenization
        
        Returns:
            Dictionary with BLEU scores and statistics
        """
        candidates = []
        references_list = []
        
        for source, target in test_data:
            # Generate translation
            if hasattr(model, 'translate'):
                translation, _ = model.translate(source)
            else:
                raise ValueError("Model must have a translate method")
            
            # Convert to text if needed
            if tokenizer is not None:
                if hasattr(tokenizer, 'decode'):
                    translation = tokenizer.decode(translation)
                    target = tokenizer.decode(target)
            
            candidates.append(translation)
            references_list.append([target])  # Single reference
        
        # Compute scores
        corpus_score = self.corpus_score(candidates, references_list)
        sentence_scores = [self.sentence_score(cand, refs) 
                          for cand, refs in zip(candidates, references_list)]
        
        return {
            'corpus_bleu': corpus_score,
            'sentence_bleu_mean': np.mean(sentence_scores),
            'sentence_bleu_std': np.std(sentence_scores),
            'sentence_scores': sentence_scores,
            'num_sentences': len(candidates)
        }

def multi_bleu_detok(candidates, references_list, lowercase=False):
    """
    Multi-BLEU score computation similar to Moses toolkit.
    
    Args:
        candidates: List of candidate translations
        references_list: List of lists of reference translations
        lowercase: Whether to lowercase before evaluation
    
    Returns:
        BLEU score
    """
    if lowercase:
        candidates = [cand.lower() for cand in candidates]
        references_list = [[ref.lower() for ref in refs] for refs in references_list]
    
    evaluator = BLEUScore(max_n=4, smoothing=True)
    return evaluator.corpus_score(candidates, references_list)

def self_bleu(candidates, n_sample=None):
    """
    Compute Self-BLEU score to measure diversity.
    
    Args:
        candidates: List of candidate sentences
        n_sample: Number of samples to use (None for all)
    
    Returns:
        Self-BLEU score
    """
    if n_sample is not None and len(candidates) > n_sample:
        import random
        candidates = random.sample(candidates, n_sample)
    
    scores = []
    for i, candidate in enumerate(candidates):
        references = candidates[:i] + candidates[i+1:]  # All other candidates as references
        if references:  # Only compute if there are references
            score = sentence_bleu(candidate, references)
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0 