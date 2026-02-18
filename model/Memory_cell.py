import torch.nn as nn
import torch
class MemoryCell(nn.module):
    # Pattern learning
    def __init__(
            self,
            input_size: int,
            pattern_size: int)-> None:
        super(MemoryCell, self).__init__()
        self.input_size = input_size
        self.pattern_size = pattern_size
        self.num_patterns = 0
        self.W = nn.Parameter(torch.zeros((self.input_size, self.pattern_size), device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))
        self.T = nn.Parameter(torch.zeros((self.input_size + self.pattern_size, self.pattern_size), device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))

    def forward(self, x):
        a = torch.matmul(x, self.W)
        s = torch.argmax(a, dim=1)
        return a,s

    # Memory chaining operations
    # k - index of the pattern to update
    def Hebbian_update(self, x, k, learning_rate=0.01):
        with torch.no_grad():
            self.W[:, k] += learning_rate * x
    
    def Temporal_chaining(self,k_cur, k_pre, learning_rate=0.01):
        with torch.no_grad():
            self.T[k_pre, k_cur] += learning_rate
    
    # Pattern management
    def new_pattern(self, x):
        with torch.no_grad():
            self.W[self.num_patterns].copy_(x)
        self.num_patterns += 1
        return self.num_patterns - 1
    
    def update_pattern(self, pattern_index, x):
        with torch.no_grad():
            self.W[pattern_index].copy_(x)
    
    def recall_pattern(self, pattern_index):
        return self.W[pattern_index]
    # prediction and generation
    def predict_next(self, s):
        s_next = torch.matmul(self.T[:self.num_patterns, :self.num_patterns].T, s)
        p_next=torch.argmax(s_next, dim=0)
        return p_next
    def generate(self,s):
        next_data = torch.matmul(self.W[:self.num_patterns].T, s)
        return next_data
    def generate_sequence(self, start_pattern_index, length):
        sequence = [start_pattern_index]
        current_pattern = start_pattern_index
        for _ in range(length - 1):
            next_pattern = self.predict_next(current_pattern)
            sequence.append(next_pattern.item())
            current_pattern = next_pattern
        return sequence
    # Step


    