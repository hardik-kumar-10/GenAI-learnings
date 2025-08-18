import numpy as np
import matplotlib.pyplot as plt
import math

class PositionalEncodingExplainer:
    """
    A detailed explanation of positional encoding in transformer models
    """
    
    def __init__(self, max_seq_len=10, embed_dim=8):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
    def explain_the_problem(self):
        """Explain WHY we need positional encoding"""
        print("🤔 THE PROBLEM: Why do we need Positional Encoding?")
        print("=" * 60)
        print()
        print("Consider these two sentences:")
        print("1. 'The cat sat on the mat'")
        print("2. 'The mat sat on the cat'")
        print()
        print("❌ WITHOUT positional encoding:")
        print("   • Both sentences have the same words")
        print("   • The model would see them as identical!")
        print("   • Word order is completely lost")
        print()
        print("✅ WITH positional encoding:")
        print("   • Each word gets its position information")
        print("   • 'cat' at position 1 ≠ 'cat' at position 5")
        print("   • The model understands word order matters")
        print()
        
        # Demonstrate with simple example
        sentence1 = ["The", "cat", "sat", "on", "mat"]
        sentence2 = ["The", "mat", "sat", "on", "cat"]
        
        print("Position mapping:")
        print("Sentence 1:")
        for i, word in enumerate(sentence1):
            print(f"   Position {i}: '{word}'")
        print()
        print("Sentence 2:")
        for i, word in enumerate(sentence2):
            print(f"   Position {i}: '{word}'")
        print()
        
    def create_simple_positional_encoding(self):
        """Create and explain the math behind positional encoding"""
        print("🧮 THE MATH: How Positional Encoding Works")
        print("=" * 60)
        print()
        print("The formula for positional encoding is:")
        print("PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))")
        print("PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
        print()
        print("Where:")
        print("• pos = position in the sequence (0, 1, 2, ...)")
        print("• i = dimension index (0, 1, 2, ...)")
        print("• d_model = embedding dimension")
        print()
        
        # Create positional encoding matrix
        pos_encoding = np.zeros((self.max_seq_len, self.embed_dim))
        
        print(f"Let's build it step by step for {self.max_seq_len} positions and {self.embed_dim} dimensions:")
        print()
        
        for pos in range(self.max_seq_len):
            print(f"Position {pos}:")
            for i in range(0, self.embed_dim, 2):
                # Calculate the angle
                angle = pos / (10000 ** (i / self.embed_dim))
                
                # Apply sin to even indices
                pos_encoding[pos, i] = math.sin(angle)
                print(f"  Dim {i} (sin): {pos_encoding[pos, i]:.4f}")
                
                # Apply cos to odd indices
                if i + 1 < self.embed_dim:
                    pos_encoding[pos, i + 1] = math.cos(angle)
                    print(f"  Dim {i+1} (cos): {pos_encoding[pos, i + 1]:.4f}")
            print()
        
        return pos_encoding
    
    def visualize_positional_encoding(self, pos_encoding):
        """Visualize the positional encoding patterns"""
        print("📊 VISUALIZATION: Positional Encoding Patterns")
        print("=" * 60)
        
        try:
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Heatmap of all positional encodings
            im1 = axes[0,0].imshow(pos_encoding.T, cmap='RdBu', aspect='auto')
            axes[0,0].set_title('Positional Encoding Matrix\n(Dimensions vs Positions)')
            axes[0,0].set_xlabel('Position in Sequence')
            axes[0,0].set_ylabel('Embedding Dimension')
            plt.colorbar(im1, ax=axes[0,0])
            
            # 2. Show patterns for specific dimensions
            axes[0,1].plot(pos_encoding[:, 0], 'b-', label='Dim 0 (sin)', linewidth=2)
            axes[0,1].plot(pos_encoding[:, 1], 'r-', label='Dim 1 (cos)', linewidth=2)
            if self.embed_dim > 2:
                axes[0,1].plot(pos_encoding[:, 2], 'g-', label='Dim 2 (sin)', linewidth=2)
            axes[0,1].set_title('Positional Encoding Values\nfor Different Dimensions')
            axes[0,1].set_xlabel('Position')
            axes[0,1].set_ylabel('Encoding Value')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # 3. Show uniqueness of each position
            positions_to_show = min(5, self.max_seq_len)
            for pos in range(positions_to_show):
                axes[1,0].plot(pos_encoding[pos, :], marker='o', label=f'Pos {pos}')
            axes[1,0].set_title('Each Position Has Unique Pattern')
            axes[1,0].set_xlabel('Embedding Dimension')
            axes[1,0].set_ylabel('Encoding Value')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # 4. Show how different frequencies work
            positions = np.arange(self.max_seq_len)
            for dim in [0, 2, 4]:
                if dim < self.embed_dim:
                    freq = 1 / (10000 ** (dim / self.embed_dim))
                    values = np.sin(positions * freq)
                    axes[1,1].plot(positions, values, label=f'Dim {dim} (freq={freq:.4f})')
            axes[1,1].set_title('Different Frequencies for Different Dimensions')
            axes[1,1].set_xlabel('Position')
            axes[1,1].set_ylabel('Sin Value')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Here's the text representation:")
            print("\nPositional Encoding Matrix:")
            print("Rows = Positions, Columns = Dimensions")
            print(pos_encoding.round(3))
    
    def demonstrate_with_words(self):
        """Show how positional encoding is added to word embeddings"""
        print("🔤 PRACTICAL EXAMPLE: Adding Position to Words")
        print("=" * 60)
        print()
        
        # Simple word embeddings (normally these would be learned)
        words = ["The", "cat", "sat", "on", "mat"]
        np.random.seed(42)
        word_embeddings = np.random.randn(len(words), self.embed_dim) * 0.5
        
        # Get positional encodings
        pos_encoding = self.create_simple_positional_encoding()
        
        print("Step-by-step process:")
        print()
        
        for i, word in enumerate(words):
            print(f"Word: '{word}' at position {i}")
            print(f"  Original embedding: {word_embeddings[i].round(3)}")
            print(f"  Positional encoding: {pos_encoding[i].round(3)}")
            
            # Add them together
            final_embedding = word_embeddings[i] + pos_encoding[i]
            print(f"  Final embedding:    {final_embedding.round(3)}")
            print()
    
    def explain_key_properties(self):
        """Explain the key properties of positional encoding"""
        print("🔑 KEY PROPERTIES: Why This Design Works")
        print("=" * 60)
        print()
        
        print("1. UNIQUENESS:")
        print("   • Each position gets a unique encoding vector")
        print("   • No two positions have the same encoding")
        print()
        
        print("2. RELATIVE POSITION:")
        print("   • The model can learn relative distances")
        print("   • PE(pos+k) has a fixed relationship to PE(pos)")
        print()
        
        print("3. SCALABILITY:")
        print("   • Works for sequences longer than training data")
        print("   • The formula works for any sequence length")
        print()
        
        print("4. SMOOTH TRANSITIONS:")
        print("   • Adjacent positions have similar encodings")
        print("   • No sudden jumps between neighboring positions")
        print()
        
        # Demonstrate relative position property
        pos_encoding = self.create_simple_positional_encoding()
        
        print("Demonstrating relative position (cosine similarity):")
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        pos_0 = pos_encoding[0]
        for i in range(min(5, self.max_seq_len)):
            sim = cosine_similarity(pos_0, pos_encoding[i])
            print(f"   Similarity between pos 0 and pos {i}: {sim:.4f}")
    
    def compare_alternatives(self):
        """Compare with alternative positional encoding methods"""
        print("🔄 ALTERNATIVES: Other Ways to Encode Position")
        print("=" * 60)
        print()
        
        print("1. LEARNED POSITIONAL EMBEDDINGS:")
        print("   • Train separate embedding matrix for positions")
        print("   • Pro: Optimized for specific task")
        print("   • Con: Fixed maximum sequence length")
        print()
        
        print("2. SIMPLE INTEGER ENCODING:")
        print("   • Just use [0, 1, 2, 3, ...] as position")
        print("   • Pro: Very simple")
        print("   • Con: Doesn't work well with different scales")
        print()
        
        print("3. SINUSOIDAL (What we use):")
        print("   • Sin/cos functions with different frequencies")
        print("   • Pro: Works for any length, smooth, unique")
        print("   • Con: More complex to understand")
        print()
        
        # Show comparison
        seq_len = min(8, self.max_seq_len)
        
        # Simple integer
        simple_pos = np.arange(seq_len).reshape(-1, 1)
        
        # Sinusoidal (our method)
        pos_encoding = self.create_simple_positional_encoding()
        
        print("Comparison for first few positions:")
        print("Position | Simple | Sinusoidal (first 3 dims)")
        print("-" * 45)
        for i in range(seq_len):
            sin_str = " ".join([f"{pos_encoding[i, j]:.2f}" for j in range(min(3, self.embed_dim))])
            print(f"   {i}     |   {i}    | {sin_str}")

def run_complete_explanation():
    """Run the complete positional encoding explanation"""
    print("🎯 POSITIONAL ENCODING: COMPLETE EXPLANATION")
    print("=" * 80)
    print()
    
    explainer = PositionalEncodingExplainer(max_seq_len=8, embed_dim=8)
    
    # Step 1: Explain the problem
    explainer.explain_the_problem()
    input("Press Enter to continue to the math explanation...")
    print("\n" + "="*80 + "\n")
    
    # Step 2: Show the math
    pos_encoding = explainer.create_simple_positional_encoding()
    input("Press Enter to see the visualization...")
    print("\n" + "="*80 + "\n")
    
    # Step 3: Visualize
    explainer.visualize_positional_encoding(pos_encoding)
    input("Press Enter to see practical example...")
    print("\n" + "="*80 + "\n")
    
    # Step 4: Practical example
    explainer.demonstrate_with_words()
    input("Press Enter to learn about key properties...")
    print("\n" + "="*80 + "\n")
    
    # Step 5: Key properties
    explainer.explain_key_properties()
    input("Press Enter to see alternatives...")
    print("\n" + "="*80 + "\n")
    
    # Step 6: Alternatives
    explainer.compare_alternatives()
    
    print("\n" + "="*80)
    print("🎉 SUMMARY: Positional Encoding in Simple Terms")
    print("="*80)
    print()
    print("Think of it like this:")
    print("• Each word gets a 'name tag' showing its position")
    print("• The name tag is a special number pattern (vector)")
    print("• We ADD this pattern to the word's meaning")
    print("• Now 'cat' at position 1 ≠ 'cat' at position 5")
    print("• The model can understand word order!")
    print()
    print("The sin/cos formula creates unique, smooth patterns")
    print("that work for any sentence length. Brilliant! 🧠✨")

if __name__ == "__main__":
    run_complete_explanation()