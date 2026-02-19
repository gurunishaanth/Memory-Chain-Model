import torch.nn as nn
import torch
class MemoryCell(nn.Module):
    # Pattern learning
    def __init__(
            self,
            input_size: int,
            pattern_size: int,
            learning_rate: float,
            Sim_Thr: float,
            Ano_Thr: float
            )-> None:
        super(MemoryCell, self).__init__()
        self.input_size = input_size
        self.pattern_size = pattern_size
        self.num_patterns = 0
        self.learning_rate = learning_rate
        self.Sim_Thr = Sim_Thr
        self.Ano_Thr = Ano_Thr
        self.prev_k = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.W = nn.Parameter(torch.zeros((self.pattern_size, self.input_size), device=device))
        self.T = nn.Parameter(torch.zeros((self.pattern_size, self.pattern_size), device=device))

    def forward(self, x):
        a = torch.matmul(x, self.W.T)
        s = torch.argmax(a, dim=1)
        return a, s

    # Memory chaining operations
    # k - index of the pattern to update
    def Hebbian_update(self, x, k):
        with torch.no_grad():
            self.W[k] += self.learning_rate * x
    
    def Temporal_chaining(self,k_cur, k_pre):
        with torch.no_grad():
            self.T[k_pre, k_cur] += self.learning_rate
    
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
    def predict_next(self, pattern_index):
        # pattern_index is an integer, get the corresponding row from T
        s_next = self.T[pattern_index, :self.num_patterns]
        p_next = torch.argmax(s_next)
        return p_next
    def generate(self,s):
        next_data = torch.matmul(self.W[:self.num_patterns], s)
        return next_data
    def generate_dream_sequence(self, start_pattern_index, length):
        sequence = [start_pattern_index]
        current_pattern = start_pattern_index
        for _ in range(length - 1):
            next_pattern = self.predict_next(current_pattern)
            sequence.append(next_pattern.item())
            current_pattern = next_pattern
        return sequence
    # Step
    def Mem_chain_step(self, x):
        if self.num_patterns == 0:
            k = self.new_pattern(x)
            self.prev_k = k
            return k, float('inf'), False
        
        a, s = self.forward(x)
        c, k = torch.max(a, dim=1)
        c = c.squeeze()
        k = k.squeeze()
        
        anomaly = c.item() < self.Ano_Thr
        similarity = c.item() < self.Sim_Thr
        
        if similarity:
            k = self.new_pattern(x)
        else:
            self.update_pattern(k.item(), x)
        
        if self.prev_k is not None:
            self.Temporal_chaining(self.prev_k, k.item())
        
        self.prev_k = k.item() if isinstance(k, torch.Tensor) else k
        return k.item() if isinstance(k, torch.Tensor) else k, c.item(), anomaly    