import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

class SimpleEmbeddingModel:
    """
    A simplified embedding model to demonstrate the core architecture concepts:
    1. Tokenization
    2. Token Embedding Layer
    3. Positional Encoding
    4. Transformer Blocks (simplified)
    5. Pooling Layer
    6. Final Dense Vector Output
    """
    
    def __init__(self, vocab_size=1000, embed_dim=128, max_seq_len=50):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Initialize embedding matrices (normally learned during training)
        np.random.seed(42)  # For reproducible results
        self.token_embeddings = np.random.randn(vocab_size, embed_dim) * 0.1
        self.positional_embeddings = self._create_positional_encoding()
        
        # Simple vocabulary for demonstration
        self.vocab = {
            '<PAD>': 0, '<UNK>': 1, 'the': 2, 'cat': 3, 'dog': 4, 'sat': 5,
            'on': 6, 'mat': 7, 'is': 8, 'running': 9, 'beautiful': 10,
            'house': 11, 'car': 12, 'tree': 13, 'book': 14, 'computer': 15
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def _create_positional_encoding(self):
        """Create sinusoidal positional encodings"""
        pos_encoding = np.zeros((self.max_seq_len, self.embed_dim))
        
        for pos in range(self.max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pos_encoding[pos, i] = math.sin(pos / (10000 ** (i / self.embed_dim)))
                if i + 1 < self.embed_dim:
                    pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** (i / self.embed_dim)))
        
        return pos_encoding
    
    def tokenize(self, text):
        """Simple tokenization (word-level)"""
        words = text.lower().split()
        tokens = []
        for word in words:
            token_id = self.vocab.get(word, self.vocab['<UNK>'])
            tokens.append(token_id)
        
        # Pad or truncate to max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens.extend([self.vocab['<PAD>']] * (self.max_seq_len - len(tokens)))
        
        return np.array(tokens)
    
    def get_token_embeddings(self, token_ids):
        """Convert token IDs to embeddings"""
        return self.token_embeddings[token_ids]
    
    def add_positional_encoding(self, token_embeds, seq_len):
        """Add positional information to token embeddings"""
        pos_embeds = self.positional_embeddings[:seq_len]
        return token_embeds[:seq_len] + pos_embeds
    
    def simple_attention(self, embeddings):
        """Simplified self-attention mechanism"""
        # In real transformers, this involves Query, Key, Value matrices
        # Here we'll just compute a simple attention-weighted average
        
        # Compute attention weights (simplified)
        attention_scores = np.dot(embeddings, embeddings.T)
        attention_weights = self.softmax(attention_scores, axis=1)
        
        # Apply attention weights
        attended_embeddings = np.dot(attention_weights, embeddings)
        return attended_embeddings, attention_weights
    
    def softmax(self, x, axis=None):
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def pool_embeddings(self, embeddings, method='mean'):
        """Pool sequence of embeddings into a single vector"""
        if method == 'mean':
            return np.mean(embeddings, axis=0)
        elif method == 'max':
            return np.max(embeddings, axis=0)
        elif method == 'cls':  # Use first token (like BERT)
            return embeddings[0]
        else:
            return np.mean(embeddings, axis=0)
    
    def encode(self, text, show_steps=False):
        """Complete encoding pipeline"""
        print(f"\n=== ENCODING: '{text}' ===")
        
        # Step 1: Tokenization
        token_ids = self.tokenize(text)
        actual_length = len(text.split())
        print(f"1. Tokenized: {[self.reverse_vocab.get(tid, f'ID_{tid}') for tid in token_ids[:actual_length]]}")
        
        # Step 2: Token Embeddings
        token_embeds = self.get_token_embeddings(token_ids)
        print(f"2. Token embeddings shape: {token_embeds.shape}")
        
        # Step 3: Add Positional Encoding
        positioned_embeds = self.add_positional_encoding(token_embeds, actual_length)
        print(f"3. With positional encoding shape: {positioned_embeds.shape}")
        
        # Step 4: Self-Attention (simplified)
        attended_embeds, attention_weights = self.simple_attention(positioned_embeds)
        print(f"4. After attention shape: {attended_embeds.shape}")
        
        # Step 5: Pooling
        final_embedding = self.pool_embeddings(attended_embeds)
        print(f"5. Final embedding shape: {final_embedding.shape}")
        print(f"   Final embedding (first 10 dims): {final_embedding[:10].round(3)}")
        
        if show_steps:
            return final_embedding, {
                'tokens': token_ids[:actual_length],
                'token_embeds': token_embeds[:actual_length],
                'positioned_embeds': positioned_embeds,
                'attention_weights': attention_weights,
                'final_embedding': final_embedding
            }
        
        return final_embedding

def demonstrate_embedding_architecture():
    """Demonstrate how embedding models work"""
    print("🔤 EMBEDDING MODEL ARCHITECTURE DEMONSTRATION")
    print("=" * 55)
    
    # Initialize model
    model = SimpleEmbeddingModel(embed_dim=64, max_seq_len=20)
    
    # Test sentences
    sentences = [
        "the cat sat on the mat",
        "the dog is running",
        "beautiful house with tree",
        "computer book"
    ]
    
    embeddings = []
    
    for sentence in sentences:
        embedding = model.encode(sentence)
        embeddings.append(embedding)
    
    # Compute similarities
    print("\n" + "=" * 55)
    print("📊 SIMILARITY ANALYSIS")
    print("=" * 55)
    
    embeddings = np.array(embeddings)
    
    # Cosine similarity matrix
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("\nCosine Similarity Matrix:")
    print("Sentences:")
    for i, sent in enumerate(sentences):
        print(f"{i}: {sent}")
    
    print("\nSimilarity scores:")
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"Sent {i} ↔ Sent {j}: {sim:.3f}")
    
    # Visualize architecture components
    print("\n" + "=" * 55)
    print("🏗️ ARCHITECTURE COMPONENTS")
    print("=" * 55)
    
    print("\n1. TOKENIZATION")
    print("   • Converts text to numerical token IDs")
    print("   • Handles unknown words with <UNK> token")
    print("   • Pads sequences to fixed length")
    
    print("\n2. TOKEN EMBEDDING LAYER")
    print(f"   • Maps each token ID to dense vector (dim: {model.embed_dim})")
    print("   • Learned during training to capture semantic meaning")
    
    print("\n3. POSITIONAL ENCODING")
    print("   • Adds position information to embeddings")
    print("   • Uses sinusoidal functions for different positions")
    
    print("\n4. TRANSFORMER BLOCKS (Simplified)")
    print("   • Self-attention: tokens attend to each other")
    print("   • Captures contextual relationships")
    
    print("\n5. POOLING LAYER")
    print("   • Combines sequence of embeddings into single vector")
    print("   • Common methods: mean, max, CLS token")
    
    print("\n6. OUTPUT")
    print(f"   • Final dense vector representation (dim: {model.embed_dim})")
    print("   • Can be used for similarity, clustering, classification")
    
    # Show detailed step-by-step for one example
    print("\n" + "=" * 55)
    print("🔍 DETAILED STEP-BY-STEP EXAMPLE")
    print("=" * 55)
    
    example_text = "the cat sat"
    final_emb, steps = model.encode(example_text, show_steps=True)
    
    return model, embeddings

def visualize_attention(model, text="the cat sat"):
    """Visualize attention patterns (if matplotlib available)"""
    try:
        final_emb, steps = model.encode(text, show_steps=True)
        attention_weights = steps['attention_weights']
        tokens = [model.reverse_vocab[tid] for tid in steps['tokens']]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(attention_weights, cmap='Blues', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.title(f'Attention Pattern for: "{text}"')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == "__main__":
    # Run the demonstration
    model, embeddings = demonstrate_embedding_architecture()
    
    print("\n" + "=" * 55)
    print("✨ KEY TAKEAWAYS")
    print("=" * 55)
    print("• Embeddings capture semantic meaning in dense vectors")
    print("• Architecture: Tokenize → Embed → Position → Attention → Pool")
    print("• Similar texts produce similar embeddings (high cosine similarity)")
    print("• Real models use much larger dimensions (768, 1024, 4096+)")
    print("• Training learns optimal embeddings for downstream tasks")