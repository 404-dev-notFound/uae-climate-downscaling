import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Custom Convolutional LSTM cell architecture.
        """
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class CNNLSTM_Downscaler(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=5):
        super(CNNLSTM_Downscaler, self).__init__()
        
        # ---- Learnable Spatial Upsampling (3x3 -> 17x17) ----
        # Stage 1: 3x3 -> 9x9 via ConvTranspose2d (stride=3, kernel=3)
        self.upsample_stage1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        # Stage 2: 9x9 -> 17x17 via ConvTranspose2d (stride=2, kernel=3, output_padding=0)
        self.upsample_stage2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Deeper Spatial Feature Extraction with BatchNorm for stable training
        self.feature_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.feature_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.feature_conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
        # Residual projection: match input channels to hidden_channels for skip connection
        self.residual_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        # Temporal Sequence Modeler
        self.conv_lstm = ConvLSTMCell(
            input_dim=hidden_channels, 
            hidden_dim=hidden_channels, 
            kernel_size=(3, 3), 
            bias=True
        )
        
        # Output mapper
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass for Spatiotemporal Downscaling.
        Args:
            x: Input sequence tensor. Expected shape: (Batch, Time, Channels, Height, Width)
               e.g., (B, Seq_Len, 5, 3, 3)
        Returns:
            Output sequence tensor of shape (Batch, Time, Channels, Target_Height, Target_Width)
               e.g., (B, Seq_Len, 5, 17, 17)
        """
        B, T, C, H, W = x.size()
        
        # Step 1: Initialize ConvLSTM hidden states matching the target grid size (17x17)
        target_h, target_w = 17, 17
        h, c = self._init_hidden(B, self.conv_lstm.hidden_dim, target_h, target_w, x.device)
        
        outputs = []
        for t in range(T):
            x_t = x[:, t, :, :, :] # (B, C, 3, 3)
            
            # Step 2: Learnable Spatial Upsampling via Transposed Convolutions
            # Stage 1: HxW -> intermediate, Stage 2: intermediate -> near-target
            x_up = self.upsample_stage1(x_t)
            x_up = self.upsample_stage2(x_up)
            
            # Adaptive resize: guarantee exactly 17x17 for any input grid (2x2, 3x3, etc.)
            if x_up.shape[2] != target_h or x_up.shape[3] != target_w:
                x_up = F.interpolate(x_up, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            # Step 3: Deeper Feature Extraction with Residual Connection
            identity = self.residual_proj(x_up)       # (B, hidden, 17, 17)
            feat = F.relu(self.bn1(self.feature_conv1(x_up)))
            feat = F.relu(self.bn2(self.feature_conv2(feat)))
            feat = self.bn3(self.feature_conv3(feat))
            feat = F.relu(feat + identity)              # Residual skip connection
            
            # Step 4: Temporal sequence modeling (Memory integration)
            h, c = self.conv_lstm(feat, (h, c))
            
            # Step 5: Final mapping back to target variables
            out_t = self.final_conv(h) # (B, 5, 17, 17)
            outputs.append(out_t)
            
        # Stack sequence dimension
        outputs = torch.stack(outputs, dim=1) # (B, T, 5, 17, 17)
        return outputs
        
    def _init_hidden(self, batch_size, hidden_dim, cur_h, cur_w, device):
        return (torch.zeros(batch_size, hidden_dim, cur_h, cur_w, device=device),
                torch.zeros(batch_size, hidden_dim, cur_h, cur_w, device=device))
