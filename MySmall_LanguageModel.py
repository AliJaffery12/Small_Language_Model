import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoding', position_encoding.unsqueeze(0))  # Shape: (1, max_len, embedding_dim)

    def forward(self, x):
        return x + self.position_encoding[:, :x.size(1), :]


# Self-Attention Mechanism Class
class SelfAttention(nn.Module):
    def __init__(self, input_dim, head_size):
        super(SelfAttention, self).__init__()
        self.head_size = head_size
        self.query = nn.Linear(input_dim, head_size, bias=False)
        self.key = nn.Linear(input_dim, head_size, bias=False)
        self.value = nn.Linear(input_dim, head_size, bias=False)
        self.output_projection = nn.Linear(head_size, input_dim, bias=False)  # New layer to project back to input_dim
        self.scale = torch.sqrt(torch.FloatTensor([head_size]))

    def forward(self, x):
        B, T, C = x.size()
        Q = self.query(x)  # (B, T, head_size)
        K = self.key(x)    # (B, T, head_size)
        V = self.value(x)  # (B, T, head_size)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, T, T)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, T, T)
        attention_output = torch.matmul(attention_weights, V)  # (B, T, head_size)
        
        attention_output = self.output_projection(attention_output)  # (B, T, input_dim)

        return attention_output, attention_weights



# Feedforward Neural Network Class
class FeedForward(nn.Module):
    def __init__(self, input_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# Transformer Block with Self-Attention, FFN, Residual Connections, and Layer Norm
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, head_size, ff_dim):
        super(TransformerBlock, self).__init__()
        self.self_attention = SelfAttention(input_dim, head_size)
        self.feed_forward = FeedForward(input_dim, ff_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Self-Attention with residual connection and layer normalization
        attention_output, _ = self.self_attention(x)
        x = self.norm1(x + attention_output)  # Add residual connection
        
        # Feedforward network with residual connection and layer normalization
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + ffn_output)  # Add residual connection
        
        return x


# Stack Multiple Transformer Blocks
class MiniTransformerModel(nn.Module):
    def __init__(self, input_dim, head_size, ff_dim, num_layers):
        super(MiniTransformerModel, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(input_dim, head_size, ff_dim) for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Final Language Model Class
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, input_dim, head_size, ff_dim, num_layers):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.pos_encoder = PositionalEncoding(input_dim)
        self.transformer = MiniTransformerModel(input_dim, head_size, ff_dim, num_layers)
        self.fc_out = nn.Linear(input_dim, vocab_size)

    def forward(self, x):
        # Get embeddings for input tokens
        x = self.embedding(x)

        # Add positional encoding to the embeddings
        x = self.pos_encoder(x)

        # Pass through transformer layers
        x = self.transformer(x)

        # Project to vocabulary size for token prediction
        logits = self.fc_out(x)  # (B, T, vocab_size)

        return logits


# Custom Dataset Class for Text Corpus
class TextDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()
        self.tokens = self.tokenize(text)

    def tokenize(self, text):
        words = text.split()
        unique_tokens = list(set(words))
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_tokens)}
        self.idx_to_word = {idx: word for idx, word in enumerate(unique_tokens)}
        self.vocab_size = len(unique_tokens)

        # Map words to indices
        token_indices = [self.word_to_idx[word] for word in words]
        return token_indices

    def __len__(self):
        return len(self.tokens) - 1  # One less than the total number of tokens

    def __getitem__(self, idx):
        # Return the input and target tokens
        input_token = self.tokens[idx]
        target_token = self.tokens[idx + 1]
        return input_token, target_token


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.view(-1, 1)  # Reshape for embedding lookup
            targets = targets.view(-1)

            optimizer.zero_grad()  # Zero gradients
            logits = model(inputs)  # Forward pass

            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets)
            total_loss += loss.item()

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')

        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}')


#Training of Model
if __name__ == "__main__":
    # Parameters
    C = 64  # Embedding size
    head_size = 16  # Head size for self-attention
    ff_dim = 256  # Feedforward network hidden dimension
    num_layers = 2  # Number of transformer layers
    num_epochs = 2  # Number of epochs for training
    batch_size = 32  # Batch size for training
    learning_rate = 0.001  # Learning rate

    # Load the dataset
    file_path = 'sample_corpus.txt'  
    dataset = TextDataset(file_path)
    vocab_size = dataset.vocab_size  # Use the size of the dataset's vocabulary
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the language model
    model = LanguageModel(vocab_size, input_dim=C, head_size=head_size, ff_dim=ff_dim, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)


def generate_text(model, start_token, max_length, dataset, temperature=1.0):
    model.eval()  # Set model to evaluation mode
    generated_tokens = [start_token]  # Start with the given token
    
    for _ in range(max_length):
        input_tensor = torch.tensor(generated_tokens).unsqueeze(0)  # Shape: (1, T)
        with torch.no_grad():  # No need to compute gradients
            logits = model(input_tensor)  # Get the output logits
        
        # Apply temperature to logits
        logits = logits[:, -1, :] / temperature  # Use the last output only
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # Shape: (vocab_size,)
        
        # Sample from the distribution
        next_token = np.random.choice(range(dataset.vocab_size), p=probabilities)
        generated_tokens.append(next_token)  # Append the predicted token
    
    return ' '.join([dataset.idx_to_word[idx] for idx in generated_tokens])  # Convert indices to words


# Choose a valid start token
known_tokens = list(dataset.word_to_idx.keys())
print("Available tokens:", known_tokens)  

# For demonstration, use the first available token
start_token = dataset.word_to_idx[known_tokens[0]]
start_token_phrase = "You shall find of the king a husband"  
if start_token_phrase in dataset.word_to_idx:
    start_token = dataset.word_to_idx[start_token_phrase]
else:
    print(f"Start token not found: {start_token_phrase}")
    start_token = dataset.word_to_idx[known_tokens[0]] 
    
generated_text = generate_text(model, start_token, max_length=50, dataset=dataset, temperature=1.0)
print("Generated Text:", generated_text)
