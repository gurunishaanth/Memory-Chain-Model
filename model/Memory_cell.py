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

    def _prepare_input(self, x):
        if isinstance(x, dict):
            tensor = torch.zeros(self.input_size, dtype=self.W.dtype, device=self.W.device)
            for idx, value in x.items():
                tensor[int(idx)] = value
            return tensor
        if isinstance(x, torch.Tensor):
            return x.to(device=self.W.device, dtype=self.W.dtype)
        raise TypeError(f"Unsupported input type: {type(x)}")

    def forward(self, x):
        x = self._prepare_input(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        a = torch.matmul(x, self.W.T)
        s = torch.argmax(a, dim=1)
        return a, s

    # Memory chaining operations
    # k - index of the pattern to update
    def Hebbian_update(self, x, k):
        with torch.no_grad():
            x = self._prepare_input(x)
            if x.dim() == 2 and x.size(0) == 1:
                x = x.squeeze(0)
            self.W[k] += self.learning_rate * x
    
    def Temporal_chaining(self,k_cur, k_pre):
        with torch.no_grad():
            self.T[k_pre, k_cur] += self.learning_rate
    
    # Pattern management
    def new_pattern(self, x):
        with torch.no_grad():
            x = self._prepare_input(x)
            if x.dim() == 2 and x.size(0) == 1:
                x = x.squeeze(0)

            if self.num_patterns < self.pattern_size:
                self.W[self.num_patterns].copy_(x)
                self.num_patterns += 1
                return self.num_patterns - 1

            # Memory is full: replace the least active pattern instead of crashing.
            replacement = torch.argmin(torch.norm(self.W, dim=1))
            self.W[replacement].copy_(x)
            return replacement.item()
    
    def update_pattern(self, pattern_index, x):
        with torch.no_grad():
            x = self._prepare_input(x)
            if x.dim() == 2 and x.size(0) == 1:
                x = x.squeeze(0)
            self.W[pattern_index].copy_(x)
    
    def recall_pattern(self, pattern_index):
        return self.W[pattern_index]

    def classify(self, dataset):
        if not isinstance(dataset, torch.Tensor):
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")
        if dataset.dim() == 1:
            dataset = dataset.unsqueeze(0)
        dataset = dataset.to(device=self.W.device, dtype=self.W.dtype)
        with torch.no_grad():
            a, s = self.forward(dataset)
            confidences, _ = torch.max(a, dim=1)
            return [
                {
                    'index': idx,
                    'pattern_id': int(pattern_id.item()),
                    'confidence': float(confidence.item()),
                }
                for idx, (pattern_id, confidence) in enumerate(zip(s, confidences))
            ]

    def group_dataset_by_pattern(self, dataset):
        if not isinstance(dataset, torch.Tensor):
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")
        if dataset.dim() == 1:
            dataset = dataset.unsqueeze(0)
        dataset = dataset.to(device=self.W.device, dtype=self.W.dtype)
        with torch.no_grad():
            _, s = self.forward(dataset)
            s = s.tolist()
        groups = {}
        for idx, pattern_id in enumerate(s):
            groups.setdefault(pattern_id, []).append(idx)
        grouped = {}
        for pattern_id, indices in groups.items():
            tensor_indices = torch.tensor(indices, dtype=torch.long, device=dataset.device)
            grouped[pattern_id] = (tensor_indices, dataset[tensor_indices])
        return grouped

    def export_temporal_chain_data(self):
        return {
            'num_patterns': self.num_patterns,
            'transition_matrix': self.T[:self.num_patterns, :self.num_patterns].detach().cpu(),
            'patterns': self.W[:self.num_patterns].detach().cpu(),
        }

    def save_temporal_chain(self, path):
        data = self.export_temporal_chain_data()
        torch.save(data, path)
        return path

    @classmethod
    def load_temporal_chain(cls, path, input_size, pattern_size, learning_rate, Sim_Thr, Ano_Thr):
        saved = torch.load(path, map_location='cpu')
        model = cls(input_size=input_size, pattern_size=pattern_size, learning_rate=learning_rate, Sim_Thr=Sim_Thr, Ano_Thr=Ano_Thr)
        num_patterns = int(saved.get('num_patterns', 0))
        model.num_patterns = min(num_patterns, model.pattern_size)
        if model.num_patterns > 0:
            model.W[:model.num_patterns].copy_(saved['patterns'][:model.num_patterns])
            model.T[:model.num_patterns, :model.num_patterns].copy_(saved['transition_matrix'][:model.num_patterns, :model.num_patterns])
        return model

    # prediction and generation
    def predict_next(self, pattern_index, randomize=False, temperature=1.0):
        if isinstance(pattern_index, torch.Tensor):
            pattern_index = pattern_index.item()
        pattern_index = int(pattern_index)
        if pattern_index < 0 or pattern_index >= self.num_patterns:
            raise IndexError(f"pattern_index {pattern_index} is out of range")

        row = self.T[pattern_index, :self.num_patterns]
        if row.numel() == 0 or torch.all(row <= 0):
            return pattern_index

        if randomize:
            logits = row / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=0)
            if torch.isfinite(probs).all() and probs.sum() > 0:
                return torch.multinomial(probs, num_samples=1).item()

        return torch.argmax(row).item()

    def generate(self, s):
        if isinstance(s, int):
            if s < 0 or s >= self.num_patterns:
                raise IndexError(f"pattern_index {s} is out of range")
            one_hot = torch.zeros(self.num_patterns, dtype=self.W.dtype, device=self.W.device)
            one_hot[s] = 1.0
            s = one_hot
        elif isinstance(s, torch.Tensor) and s.dim() == 0:
            s = torch.nn.functional.one_hot(s.long(), num_classes=self.num_patterns).to(dtype=self.W.dtype, device=self.W.device)
        elif isinstance(s, torch.Tensor) and s.dim() == 1 and s.size(0) == 1:
            s = s.squeeze(0)

        next_data = torch.matmul(self.W[:self.num_patterns].T, s)
        return next_data

    def generate_dream_sequence(self, start_pattern_index, length, randomize=False, temperature=1.0):
        if self.num_patterns == 0:
            return []
        sequence = [int(start_pattern_index)]
        current_pattern = int(start_pattern_index)
        for _ in range(length - 1):
            next_pattern = self.predict_next(current_pattern, randomize=randomize, temperature=temperature)
            sequence.append(next_pattern)
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
            k = k.item()
            self.update_pattern(k, x)
        
        if self.prev_k is not None:
            self.Temporal_chaining(self.prev_k, k)
        
        self.prev_k = k
        return k, c.item(), anomaly    