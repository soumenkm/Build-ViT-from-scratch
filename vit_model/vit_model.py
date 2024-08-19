import torch

class ViTConfig:
    def __init__(self, config: dict):
        self.W = config["image_size"]
        self.C = config["num_channels"]
        self.d = config["hidden_dim"]
        self.P = config["patch_size"]
        self.T = int((self.W / self.P) ** 2) + 1
        
class PatchEmbedding(torch.nn.Module):
    def __init__(self, device: torch.device, config: ViTConfig):
        super(PatchEmbedding, self).__init__()
        self.device = device
        self.vc = config
        self.conv_proj = torch.nn.Conv2d(in_channels=self.vc.C, 
                                         out_channels=self.vc.d, 
                                         kernel_size=(self.vc.P, self.vc.P),
                                         stride=(self.vc.P, self.vc.P),
                                         bias=True)
        self.cls_emd = torch.nn.Parameter(data=torch.zeros(size=(1, 1, self.vc.d)))
    
    def _validate_input(self, x: torch.tensor) -> None:
        if not (x.dim() == 4):
            raise ValueError(f"Input must have of rank 4! Current rank is {x.dim()}")
        b, C, W, W = x.shape
        if not (C == self.vc.C and W == self.vc.W):
            raise ValueError(f"Input must have shape ({b}, {self.vc.C}, {self.vc.W}, {self.vc.W})! Current shape is {x.shape}")

    def forward(self, x: torch.tensor) -> torch.tensor:
        """x.shape = (b, C, W, W) 
        z.shape = (b, T, d)"""
        self._validate_input(x)
        y = self.conv_proj(x) # (b, d, \sqrt(T-1), \sqrt(T-1))
        y = torch.flatten(y, start_dim=2) # (b, d, T-1)
        y = torch.transpose(y, dim0=1, dim1=2) # (b, T-1, d)
        y_cls = self.cls_emd.expand(y.shape[0], -1, -1) # (1, 1, d) -> (b, 1, d)
        z = torch.cat([y_cls, y], dim=1) # (b, T, d)
        return z

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, config: ViTConfig):
        super(PositionalEmbedding, self).__init__()
        self.vc = config
        
        
def main():
    config = {
        "image_size": 224,
        "num_channels": 3,
        "patch_size": 16,
        "hidden_dim": 768,
    }
    input_x = torch.rand(size=(16, 3, 224, 224))
    model = PatchEmbedding(device="cpu", config=ViTConfig(config))
    output_y = model(input_x)
    print(output_y.shape)
    
if __name__ == "__main__":
    main()