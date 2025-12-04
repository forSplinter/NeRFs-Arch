#This can be used as a baseline for any NeRF implementation
#This implementation is inspired by the original NeRF paper with some simplifications and no keras 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.encoding import IntegratedPositionalEncoding, PositionalEncoding, HashEncoding


class MLP(nn.Module):
    def __init__(self, D: int, W: int, input_ch: int, input_ch_views: int, output_ch: int, 
                 skips: Optional[list[int]] = None, use_viewdirs: bool = False) -> None:
        """_summary_

        Args:
            D (int): _description_
            W (int): _description_
            input_ch (int): _description_
            input_ch_views (int): _description_
            output_ch (int): _description_
            skips (Optional[list[int]], optional): _description_. Defaults to None.
            use_viewdirs (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips if skips is not None else []
        self.use_viewdirs = use_viewdirs
        
        # Point processing layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) 
             for i in range(D - 1)]
        )
        
        if use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)
            self.views_linears = nn.ModuleList([nn.Linear(W + input_ch_views, W // 2)])
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
       
        # Split input into points and view directions
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        
        # Process positions through MLP
        h = input_pts
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)
        
        if self.use_viewdirs:
            # Predict density
            alpha = self.alpha_linear(h)
            
            # Extract geometric features
            feature = self.feature_linear(h)
            
            # Combine with view directions
            h = torch.cat([feature, input_views], dim=-1)
            
            # Process view-dependent features
            for layer in self.views_linears:
                h = layer(h)
                h = F.relu(h)
            
            # Predict RGB
            rgb = self.rgb_linear(h)
            
            # Concatenate RGB and density
            outputs = torch.cat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)
        
        return outputs

class MipMLP(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 4, net_depth: int = 8, net_width: int = 256, skips: Optional[list[int]] = None,
        viewdirs: bool = True, use_embed: bool = True, multires: int = 10, multires_views: int = 4, netchunk: int = 1024 * 64) -> None:
        super().__init__()
        self.viewdirs = viewdirs
        self.chunk = netchunk 
        
        if skips is None:
            skips = [4]
        
        if use_embed:
            from utils.encoding import IntegratedPositionalEncoding, PositionalEncoding
            self.embedder = IntegratedPositionalEncoding(
                L=multires, num_freqs=multires, include_input=True
            )
            input_ch = input_dim + 2 * multires * input_dim
            
            if viewdirs:
                self.embeddirs = PositionalEncoding(L=multires_views, include_input=True)
                input_ch_views = input_dim + 2 * multires_views * input_dim
            else:
                self.embeddirs = None
                input_ch_views = 0
        else:
            self.embedder = lambda x, x_cov: x
            self.embeddirs = lambda x: x if viewdirs else None
            input_ch = input_dim
            input_ch_views = input_dim if viewdirs else 0
        
        self.mlp = MLP( D=net_depth, W=net_width, input_ch=input_ch, input_ch_views=input_ch_views, output_ch=output_dim, skips=skips, use_viewdirs=viewdirs)
    
    def set_embedder(self, pos_enc: IntegratedPositionalEncoding, dir_enc: Optional[PositionalEncoding] = None) -> None:
        self.embedder = pos_enc
        self.embeddirs = dir_enc
    
    def forward(self, x: torch.Tensor, x_cov: torch.Tensor, viewdirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (..., 3) positions
            x_cov: (..., 3) covariance diagonal
            viewdirs: (..., 3) view directions
        
        Returns:
            (..., 4) RGB + density
        """
        if self.embedder is None:
            raise ValueError("Position embedder has not been set.")
        
        # Store original shape
        input_shape = x.shape[:-1]
        
        # Flatten
        x_flat = x.reshape(-1, x.shape[-1])
        x_cov_flat = x_cov.reshape(-1, x_cov.shape[-1])
        
        if viewdirs is not None:
            viewdirs_flat = viewdirs.reshape(-1, viewdirs.shape[-1])
            assert x_flat.shape[0] == viewdirs_flat.shape[0], \
                f"Mismatch: {x_flat.shape[0]} pts vs {viewdirs_flat.shape[0]} dirs"
        else:
            viewdirs_flat = None
        # Process in chunks
        output_chunks = []
        for i in range(0, x_flat.shape[0], self.chunk):
            end = min(i + self.chunk, x_flat.shape[0])
            embedded = self.embedder(x_flat[i:end], x_cov_flat[i:end])

            if self.viewdirs and viewdirs_flat is not None:
                if self.embeddirs is None:
                    raise ValueError("Direction embedder not set.")
                
                from utils.encoding import IntegratedPositionalEncoding
                if isinstance(self.embeddirs, IntegratedPositionalEncoding):
                    zero_cov = torch.zeros_like(viewdirs_flat[i:end])
                    embedded_dirs = self.embeddirs(viewdirs_flat[i:end], zero_cov)
                else:
                    embedded_dirs = self.embeddirs(viewdirs_flat[i:end])
                
                embedded = torch.cat([embedded, embedded_dirs], dim=-1)
            
            # Forward through MLP
            chunk_output = self.mlp(embedded)
            output_chunks.append(chunk_output)
        
        # Concatenate chunks
        output_flat = torch.cat(output_chunks, dim=0)
        
        # Unflatten to original shape
        output_shape = tuple(input_shape) + (output_flat.shape[-1],)
        return output_flat.reshape(output_shape)

class InstantNGP(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 4, net_depth: int = 2, net_width: int = 64, geo_feat_dim: int = 15, 
        viewdirs: bool = True, use_embed: bool = True, num_levels: int = 16, features_per_level: int = 2, multires_views: int = 4,
        netchunk: int = 1024 * 64) -> None:
        """
        InstantNGP with multi-resolution hash encoding
        
        Args:
            input_dim: Input dimension (3 for xyz)
            output_dim: Output dimension (4 for RGB + density)
            net_depth: Depth of density network
            net_width: Width of density network
            geo_feat_dim: Dimension of geometric features
            viewdirs: Whether to use view directions
            use_embed: Whether to use encoders (True) or identity (False)
            num_levels: Number of hash encoding levels
            features_per_level: Features per level in hash encoding
            multires_views: Number of frequencies for direction encoding
            netchunk: Chunk size for processing (avoid OOM)
        """
        super().__init__()
        self.viewdirs = viewdirs
        self.geo_feat_dim = geo_feat_dim
        self.chunk = netchunk 
        
        hash_input_ch = input_dim + num_levels * features_per_level
        dir_input_ch = input_dim + 2 * multires_views * input_dim if viewdirs else 0
        
        if use_embed:
            from utils.encoding import HashEncoding, PositionalEncoding
            
            self.hash_encoder = HashEncoding(
                L=num_levels,
                F=features_per_level,
                input_dim=input_dim,
                include_input=True,
                trainable=True,
                base_resolution=16,
                per_level_scale=1.5,
                log2_hashmap_size=19
            )
            
            if viewdirs:
                self.dir_encoder = PositionalEncoding(L=multires_views, include_input=True)
            else:
                self.dir_encoder = None
        else:
            # Identity encoders
            self.hash_encoder = lambda x: x
            self.dir_encoder = lambda x: x if viewdirs else None
            hash_input_ch = input_dim
            dir_input_ch = input_dim if viewdirs else 0
        
        # Density network
        density_layers = []
        in_dim = hash_input_ch
        for i in range(net_depth):
            density_layers.append(nn.Linear(in_dim, net_width))
            density_layers.append(nn.ReLU(inplace=True))
            in_dim = net_width
        
        self.density_net = nn.Sequential(*density_layers)
        self.density_output = nn.Linear(net_width, 1 + geo_feat_dim)
        
        # Color network
        if viewdirs:
            self.color_net = nn.Sequential(
                nn.Linear(geo_feat_dim + dir_input_ch, net_width),
                nn.ReLU(inplace=True),
                nn.Linear(net_width, 3),
                nn.Sigmoid(),
            )
        else:
            self.color_net = nn.Sequential(
                nn.Linear(geo_feat_dim, 3),
                nn.Sigmoid()
            )
        
        self.density_activation = nn.ReLU(inplace=True)
    
    def set_encoder( self, hash_enc: HashEncoding, dir_enc: Optional[PositionalEncoding] = None) -> None:
        """Override default encoders with custom ones"""
        self.hash_encoder = hash_enc
        self.dir_encoder = dir_enc
    
    def forward(self, x: torch.Tensor, viewdirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through InstantNGP
        
        Args:
            x: (..., 3) 3D positions in [0, 1] (any shape ending in 3)
            viewdirs: (..., 3) view directions (optional)
        
        Returns:
            (..., 4) RGB + density with same shape prefix as input
        """
        if self.hash_encoder is None:
            raise ValueError("Hash encoder has not been set.")
        
        input_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        if viewdirs is not None:
            viewdirs_flat = viewdirs.reshape(-1, viewdirs.shape[-1])
            assert x_flat.shape[0] == viewdirs_flat.shape[0], \
                f"Mismatch: {x_flat.shape[0]} pts vs {viewdirs_flat.shape[0]} dirs"
        else:
            viewdirs_flat = None
        
        output_chunks = []
        for i in range(0, x_flat.shape[0], self.chunk):
            end = min(i + self.chunk, x_flat.shape[0])
            
            # HashEncoding expects (B, N, 3) or (N, 3)
            x_chunk = x_flat[i:end]
            if x_chunk.dim() == 2:
                h = self.hash_encoder(x_chunk.unsqueeze(0)).squeeze(0)
            else:
                h = self.hash_encoder(x_chunk)
            
            # Predict density and geometric features
            density_feat = self.density_net(h)
            density_geo = self.density_output(density_feat)
            density = self.density_activation(density_geo[..., :1])
            geo_feat = density_geo[..., 1:]
            
            # Predict color
            if self.viewdirs and viewdirs_flat is not None:
                if self.dir_encoder is not None:
                    encoded_dirs = self.dir_encoder(viewdirs_flat[i:end])
                else:
                    encoded_dirs = viewdirs_flat[i:end]
                color_input = torch.cat([geo_feat, encoded_dirs], dim=-1)
            else:
                color_input = geo_feat
            
            rgb = self.color_net(color_input)
            chunk_output = torch.cat([rgb, density], dim=-1)
            output_chunks.append(chunk_output)
        
        # Concatenate chunks
        output_flat = torch.cat(output_chunks, dim=0)
        
        output_shape = tuple(input_shape) + (output_flat.shape[-1],)
        return output_flat.reshape(output_shape)