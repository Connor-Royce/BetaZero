
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with skip connection.
    """
    
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=num_channels)
        
        self.conv2 = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=num_channels)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, num_channels, 9, 9)
        
        Returns:
            Output tensor after residual block and skip connection (batch_size, num_channels, 9, 9)
        """
        # Save the input for the skip connection
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        
        out = F.relu(out)
        
        return out

class UltimateTTTNet(nn.Module):
    """
    Neural network that takes a 5x9x9 board state and outputs:
    - Policy: probability distribution over 81 moves
    - Value: position evaluation in [-1, 1]
    """
    
    def __init__(self, num_channels=64, num_res_blocks=4):
        """
        Args:
            num_channels: Number of filters/channels (default 64)
            num_res_blocks: Number of residual blocks (default 4)
        """
        super(UltimateTTTNet, self).__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(in_channels=5, out_channels=num_channels, 
                                    kernel_size=3, padding=1, stride = 1)
        self.bn_input = nn.BatchNorm2d(num_features=num_channels)
        
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for res_block in range(num_res_blocks)])

        # Reduce channels then flatten to 81 outputs

        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=1)

        self.policy_bn = nn.BatchNorm2d(num_features=32)

        # Fully connected layer: flatten 32×9×9 → 81 outputs
        self.policy_fc = nn.Linear(in_features=2592, out_features=81)

        # 1x1 conv to reduce channels (num_channels → 32)
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=1)

        # Batch norm for 32 channels
        self.value_bn = nn.BatchNorm2d(num_features=32)

        # First fully connected layer (flatten 32×9×9 → 256)
        self.value_fc1 = nn.Linear(in_features=2592, out_features=256)

        # Second fully connected layer (256 → 1)
        self.value_fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, 5, 9, 9)
        
        Returns:
            policy_logits: shape (batch_size, 81)
            value: shape (batch_size, 1)
        """
        # Initial convolution
        out = self.conv_input(x)
        out = self.bn_input(out)
        out = F.relu(out)
        
        # Pass through residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)
        
        policy = self.policy_conv(out)    # (batch, 32, 9, 9)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        
        policy = policy.view(policy.size(0), -1)
        
        # through fully connected layer
        policy_logits = self.policy_fc(policy)  # (batch, 81)
        
        value = self.value_conv(out)      # (batch, 32, 9, 9)
        value = self.value_bn(value)
        value = F.relu(value)     
        #Flatten value
        value = value.view(value.size(0), -1)  # (batch, 32*9*9)  
        # First FC layer with ReLU
        value = self.value_fc1(value)
        value = F.relu(value) 
        # Second FC layer
        value = self.value_fc2(value) 
        #Apply tanh to squash value to [-1, 1]
        value = torch.tanh(value)
        
        return policy_logits, value
    
    def predict(self, state):
        """
        Args:
            (5x9x9) Numpy array
        Returns:
            numpy policy and value
        """
        # without GPU: state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.conv_input.weight.device)
        with torch.no_grad():
            policy_logits, value = self.forward(state)
        policy = torch.softmax(policy_logits, dim=1)
        return policy.squeeze(0).cpu().numpy(), value.squeeze(0).cpu().numpy()

