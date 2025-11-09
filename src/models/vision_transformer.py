import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class PatchEmbedding1D(nn.Module):
    """Преобразование 1D сигнала в последовательность патчей"""
    
    def __init__(self, seq_length, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (seq_length + patch_size - 1) // patch_size  # ceil division
        
        self.proj = nn.Conv1d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (batch_size, channels, seq_length)
        x = self.proj(x)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class PositionalEncoding1D(nn.Module):
    """Позиционное кодирование для 1D последовательности"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Многоголовое внимание"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Слой трансформера"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EEGVisionTransformer(nn.Module):
    """Vision Transformer для классификации EEG сигналов"""
    
    def __init__(self, input_channels=8, seq_length=250, patch_size=25, 
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, 
                 dropout=0.1, num_classes=2):
        super().__init__()
        
        self.patch_embed = PatchEmbedding1D(seq_length, patch_size, input_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Класс токен и позиционное кодирование
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding1D(embed_dim, num_patches + 1)
        self.dropout = nn.Dropout(dropout)
        
        # Трансформер энкодер
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Голова классификации
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_custom)
        
    def _init_weights_custom(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
                
    def forward(self, x):
        # x: (batch_size, channels, seq_length)
        batch_size = x.shape[0]
        
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches+1, embed_dim)
        
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        
        cls_output = x[:, 0]
        
        return self.head(cls_output)

# Альтернативная версия с канальным вниманием
class ChannelWiseAttention(nn.Module):
    """Внимание по каналам перед ViT"""
    
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels, num_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch_size, channels, seq_length)
        attention_weights = self.attention(x).unsqueeze(2)
        return x * attention_weights

class EnhancedEEGViT(nn.Module):
    """Улучшенный ViT с канальным вниманием"""
    
    def __init__(self, input_channels=8, seq_length=250, patch_size=25, 
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, 
                 dropout=0.1, num_classes=2):
        super().__init__()
        
        self.channel_attention = ChannelWiseAttention(input_channels)
        self.vit = EEGVisionTransformer(input_channels, seq_length, patch_size,
                                      embed_dim, depth, num_heads, mlp_ratio,
                                      dropout, num_classes)
        
    def forward(self, x):
        x = self.channel_attention(x)
        return self.vit(x)