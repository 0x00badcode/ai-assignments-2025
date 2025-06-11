"""
Complete example of Neural Machine Translation using LSTM from scratch.
Demonstrates training and evaluation of the Seq2Seq model.
"""
import numpy as np
import sys
import os
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.seq2seq import Seq2SeqModel
from src.training.optimizer import create_optimizer, create_scheduler, GradientClipper
from src.evaluation.bleu import BLEUScore
from src.core.utils import pad_sequences, create_mask

class SimpleTokenizer:
    """Simple tokenizer for demonstration."""
    
    def __init__(self):
        self.word_to_id = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.id_to_word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.vocab_size = 4
    
    def fit(self, sentences):
        """Build vocabulary from sentences."""
        for sentence in sentences:
            for word in sentence.split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size
                    self.id_to_word[self.vocab_size] = word
                    self.vocab_size += 1
    
    def encode(self, sentence):
        """Convert sentence to token IDs."""
        tokens = sentence.split()
        return [self.word_to_id.get(word, 3) for word in tokens]  # 3 is <UNK>
    
    def decode(self, token_ids):
        """Convert token IDs to sentence."""
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(token_id, '<UNK>')
            if word in ['<PAD>', '<START>']:
                continue
            if word == '<END>':
                break
            words.append(word)
        return ' '.join(words)

def create_toy_dataset():
    """Create a toy English-French translation dataset."""
    en_sentences = [
        "hello world",
        "how are you",
        "good morning",
        "thank you",
        "see you later",
        "i love you",
        "what is your name",
        "where are you from",
        "nice to meet you",
        "have a good day",
        "i am fine",
        "what time is it",
        "how much does it cost",
        "can you help me",
        "i don't understand",
        "speak slowly please",
        "where is the bathroom",
        "i would like coffee",
        "the weather is nice",
        "i am learning french"
    ]
    
    fr_sentences = [
        "bonjour monde",
        "comment allez vous",
        "bonjour",
        "merci",
        "a bientot",
        "je vous aime",
        "quel est votre nom",
        "d ou venez vous",
        "ravi de vous rencontrer",
        "bonne journee",
        "je vais bien",
        "quelle heure est il",
        "combien ca coute",
        "pouvez vous m aider",
        "je ne comprends pas",
        "parlez lentement s il vous plait",
        "ou sont les toilettes",
        "je voudrais du cafe",
        "il fait beau",
        "j apprends le francais"
    ]
    
    return list(zip(en_sentences, fr_sentences))

def prepare_data(dataset, src_tokenizer, tgt_tokenizer, max_length=20):
    """Prepare dataset for training."""
    src_sequences = []
    tgt_sequences = []
    
    for src_sent, tgt_sent in dataset:
        # Encode source
        src_ids = src_tokenizer.encode(src_sent)
        if len(src_ids) <= max_length:
            src_sequences.append(src_ids)
            
            # Encode target with START and END tokens
            tgt_ids = [1] + tgt_tokenizer.encode(tgt_sent) + [2]  # <START> ... <END>
            if len(tgt_ids) <= max_length + 1:
                tgt_sequences.append(tgt_ids)
            else:
                src_sequences.pop()  # Remove corresponding source
    
    return src_sequences, tgt_sequences

def create_batches(src_sequences, tgt_sequences, batch_size=4):
    """Create batches from sequences."""
    num_samples = len(src_sequences)
    indices = np.random.permutation(num_samples)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_src = [src_sequences[i] for i in batch_indices]
        batch_tgt = [tgt_sequences[i] for i in batch_indices]
        
        # Pad sequences
        batch_src_padded = pad_sequences(batch_src, padding='post', value=0)
        batch_tgt_padded = pad_sequences(batch_tgt, padding='post', value=0)
        
        # Create masks
        src_mask = create_mask(batch_src_padded, pad_value=0)
        tgt_mask = create_mask(batch_tgt_padded, pad_value=0)
        
        yield batch_src_padded, batch_tgt_padded, src_mask, tgt_mask

def train_model():
    """Main training function."""
    print("Creating toy dataset...")
    dataset = create_toy_dataset()
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Create tokenizers
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    
    # Fit tokenizers on training data
    src_sentences = [src for src, _ in train_data]
    tgt_sentences = [tgt for _, tgt in train_data]
    
    src_tokenizer.fit(src_sentences)
    tgt_tokenizer.fit(tgt_sentences)
    
    print(f"Source vocab size: {src_tokenizer.vocab_size}")
    print(f"Target vocab size: {tgt_tokenizer.vocab_size}")
    
    # Prepare training data
    train_src, train_tgt = prepare_data(train_data, src_tokenizer, tgt_tokenizer)
    test_src, test_tgt = prepare_data(test_data, src_tokenizer, tgt_tokenizer)
    
    print(f"Training sequences: {len(train_src)}")
    print(f"Test sequences: {len(test_src)}")
    
    # Create model
    model_config = {
        'src_vocab_size': src_tokenizer.vocab_size,
        'tgt_vocab_size': tgt_tokenizer.vocab_size,
        'embedding_dim': 64,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout_rate': 0.1,
        'recurrent_dropout_rate': 0.1,
        'encoder_type': 'bilstm',
        'attention_type': 'bahdanau',
        'attention_size': 64
    }
    
    print("Creating model...")
    model = Seq2SeqModel(**model_config)
    
    # Create optimizer
    optimizer = create_optimizer('adam', learning_rate=0.001)
    scheduler = create_scheduler(optimizer, 'step', step_size=500, gamma=0.5)
    clipper = GradientClipper(max_norm=5.0)
    
    # Training parameters
    num_epochs = 50
    batch_size = 4
    print_every = 10
    eval_every = 25
    
    # Training loop
    print("\nStarting training...")
    step = 0
    best_bleu = 0.0
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Create batches
        batches = list(create_batches(train_src, train_tgt, batch_size))
        
        for batch_src, batch_tgt, src_mask, tgt_mask in batches:
            step += 1
            
            # Forward pass and compute loss
            loss, gradients = model.train_step(
                batch_src, batch_tgt[:, :-1],  # Input: exclude last token
                src_mask, tgt_mask[:, :-1],    # Target: exclude first token
                label_smoothing=0.1
            )
            
            # Clip gradients
            gradients = clipper.clip(gradients)
            
            # Update parameters
            parameters = model.get_parameters()
            optimizer.update(parameters, gradients)
            
            # Update learning rate
            scheduler.step()
            
            epoch_losses.append(loss)
            
            # Print progress
            if step % print_every == 0:
                avg_loss = np.mean(epoch_losses[-print_every:])
                lr = optimizer.learning_rate
                print(f"Step {step:4d} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
        
        # Evaluation
        if (epoch + 1) % eval_every == 0:
            print(f"\nEvaluating at epoch {epoch + 1}...")
            
            # Evaluate on test set
            bleu_score = evaluate_model(model, test_src, test_tgt, src_tokenizer, tgt_tokenizer)
            
            print(f"BLEU Score: {bleu_score:.4f}")
            
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                print(f"New best BLEU score: {best_bleu:.4f}")
        
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1:2d} | Average Loss: {avg_epoch_loss:.4f}")
    
    print(f"\nTraining completed. Best BLEU score: {best_bleu:.4f}")
    
    # Final evaluation with examples
    print("\nTranslation examples:")
    model.eval_mode()
    
    for i, (src_seq, tgt_seq) in enumerate(zip(test_src[:5], test_tgt[:5])):
        src_text = src_tokenizer.decode(src_seq)
        tgt_text = tgt_tokenizer.decode(tgt_seq[1:-1])  # Remove START and END
        
        # Translate
        src_batch = np.array([src_seq])
        translations, _ = model.translate(src_batch, max_length=20)
        pred_text = tgt_tokenizer.decode(translations[0])
        
        print(f"Source: {src_text}")
        print(f"Target: {tgt_text}")
        print(f"Prediction: {pred_text}")
        print()

def evaluate_model(model, test_src, test_tgt, src_tokenizer, tgt_tokenizer):
    """Evaluate model using BLEU score."""
    model.eval_mode()
    
    candidates = []
    references = []
    
    for src_seq, tgt_seq in zip(test_src, test_tgt):
        # Translate
        src_batch = np.array([src_seq])
        translations, _ = model.translate(src_batch, max_length=20)
        
        # Convert to text
        pred_text = tgt_tokenizer.decode(translations[0])
        ref_text = tgt_tokenizer.decode(tgt_seq[1:-1])  # Remove START and END
        
        candidates.append(pred_text)
        references.append([ref_text])
    
    # Compute BLEU score
    evaluator = BLEUScore(smoothing=True)
    bleu_score = evaluator.corpus_score(candidates, references)
    
    model.train_mode()
    return bleu_score

def demonstrate_attention():
    """Demonstrate attention visualization (simplified)."""
    print("\nDemonstrating attention mechanisms...")
    
    # Create simple example
    dataset = [("hello world", "bonjour monde")]
    
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    
    src_tokenizer.fit(["hello world good morning"])
    tgt_tokenizer.fit(["bonjour monde bon matin"])
    
    # Prepare data
    src_seq = src_tokenizer.encode("hello world")
    tgt_seq = [1] + tgt_tokenizer.encode("bonjour monde") + [2]
    
    # Create model with attention
    model = Seq2SeqModel(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        embedding_dim=32,
        hidden_size=64,
        num_layers=1,
        attention_type='bahdanau'
    )
    
    # Forward pass to get attention weights
    src_batch = np.array([src_seq])
    tgt_batch = np.array([tgt_seq])
    
    logits, attention_weights = model.forward(src_batch, tgt_batch)
    
    if attention_weights is not None:
        print("Attention weights shape:", attention_weights.shape)
        print("Source tokens:", [src_tokenizer.id_to_word[id] for id in src_seq])
        print("Target tokens:", [tgt_tokenizer.id_to_word[id] for id in tgt_seq])
        print("Attention matrix (target -> source):")
        print(attention_weights[0])  # First batch item
    else:
        print("No attention weights (attention not enabled)")

if __name__ == "__main__":
    print("LSTM-based Neural Machine Translation Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Train model
    train_model()
    
    # Demonstrate attention
    demonstrate_attention()
    
    print("\nDemo completed!") 