import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPSILON = 1e-10


def _cross_squared_distance_matrix(x, y):
    """
    Args:
        x: [batch_size, n, d] float tensor
        y: [batch_size, m, d] float tensor
    Returns:
        squared_dists: [batch_size, n, m] float tensor
    """
    x_norm_squared = torch.sum(x**2, dim=-1)  # bs, n, d -> bs, n
    y_norm_squared = torch.sum(y**2, dim=-1)  # bs, m, d -> bs, m
    
    x_norm_squared_tile = x_norm_squared.unsqueeze(-1)  # [b, n, 1]
    y_norm_squared_tile = y_norm_squared.unsqueeze(-2)  # [b, 1, m]
    
    x_y_transpose = torch.matmul(x, y.transpose(-1, -2))  # [bs, n, 1] * [bs, 1, m] -> [bs, n, m]
    
    squared_dists = x_norm_squared_tile - 2 * x_y_transpose + y_norm_squared_tile
    return squared_dists


def _pairwise_squared_distance_matrix(x):
    """
    Args:
        x: [batch_size, n, d] float tensor
    Returns:
        squared_dists: [batch_size, n, n] float tensor
    """
    x_x_transpose = torch.matmul(x, x.transpose(-1, -2))
    x_norm_squared = torch.diagonal(x_x_transpose, dim1=-2, dim2=-1)
    x_norm_squared_tile = x_norm_squared.unsqueeze(2)
    
    squared_dists = x_norm_squared_tile - 2 * x_x_transpose + x_norm_squared_tile.transpose(-1, -2)
    return squared_dists


def _phi(r, order):
    """
    Coordinate-wise nonlinear function for defining interpolation order (RBF kernel function of thin plate splines)
    Args:
        r: input distance
        order: interpolation order
    Returns:
        phi_k: coordinate-wise evaluation at r
    """
    if order == 1:
        r = torch.clamp(r, min=EPSILON)
        return torch.sqrt(r)
    elif order == 2:
        # Thin Plate Spline：phi(r) = 0.5 * r * log(r)
        return 0.5 * r * torch.log(torch.clamp(r, min=EPSILON))
    elif order == 4:
        return 0.5 * r**2 * torch.log(torch.clamp(r, min=EPSILON))
    elif order % 2 == 0:
        r = torch.clamp(r, min=EPSILON)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.clamp(r, min=EPSILON)
        return torch.pow(r, 0.5 * order)


def _solve_interpolation(train_points, train_values, order, regularization_weight=0.0):
    """
    Solve interpolation coefficients
    Args:
        train_points: [b, n, d] interpolation centers
        train_values: [b, n, k] function values
        order: interpolation order
        regularization_weight: regularization weight
    Returns:
        w: [b, n, k] weights for each interpolation center
        v: [b, d+1, k] weights for each input dimension (including bias term)
    """
    b, n, d = train_points.shape
    k = train_values.shape[-1]
    matrix_a = _phi(_pairwise_squared_distance_matrix(train_points), order)  # [b, n, n]
    
    if regularization_weight > 0:
        batch_identity_matrix = torch.eye(n, dtype=train_points.dtype, device=train_points.device).unsqueeze(0).expand(b, -1, -1)
        matrix_a += regularization_weight * batch_identity_matrix
    
    # Add bias term to features
    ones = torch.ones_like(train_points[..., :1])
    matrix_b = torch.cat([train_points, ones], dim=2)  # [b, n, d+1]
    
    # Build left and right blocks
    left_block = torch.cat([matrix_a, matrix_b.transpose(-1, -2)], dim=1)  # [b, n+d+1, n]
    
    lhs_zeros = torch.zeros(b, d+1, d+1, dtype=train_points.dtype, device=train_points.device)
    right_block = torch.cat([matrix_b, lhs_zeros], dim=1)  # [b, n+d+1, d+1]
    lhs = torch.cat([left_block, right_block], dim=2)  # [b, n+d+1, n+d+1]
    
    rhs_zeros = torch.zeros(b, d+1, k, dtype=train_points.dtype, device=train_points.device)
    rhs = torch.cat([train_values, rhs_zeros], dim=1)  # [b, n+d+1, k]
    
    # Solve linear system
    try:
        w_v = torch.linalg.solve(lhs, rhs)
    except:
        # Fall back to old version of solve (for PyTorch < 1.9)
        w_v = torch.solve(rhs, lhs)[0]
    
    w = w_v[:, :n, :]
    v = w_v[:, n:, :]
    
    return w, v


def _apply_interpolation(query_points, train_points, w, v, order):
    """
    Apply multiharmonic interpolation model to data
    Args:
        query_points: [b, m, d] x values to evaluate interpolation
        train_points: [b, n, d] x values as interpolation centers
        w: [b, n, k] weights for each interpolation center
        v: [b, d+1, k] weights for each input dimension
        order: interpolation order
    Returns:
        Multiharmonic interpolation evaluated at query points
    """
    # Calculate the contribution of the RBF term
    pairwise_dists = _cross_squared_distance_matrix(query_points, train_points)
    phi_pairwise_dists = _phi(pairwise_dists, order)
    rbf_term = torch.matmul(phi_pairwise_dists, w)
    
    # Calculate the contribution of the linear term
    # Add bias term to query points
    query_points_pad = torch.cat([
        query_points,
        torch.ones_like(query_points[..., :1])
    ], dim=2)
    linear_term = torch.matmul(query_points_pad, v)
    
    return rbf_term + linear_term


def interpolate_spline(train_points, train_values, query_points, order, regularization_weight=0.0):
    """
    Use multiharmonic interpolation for signal interpolation (PyTorch version)
    Args:
        train_points: [batch_size, n, d] training points
        train_values: [batch_size, n, k] training values  
        query_points: [batch_size, m, d] query points
        order: interpolation order (2 = thin plate spline)
        regularization_weight: regularization weight
    Returns:
        [batch_size, m, k] query values
    """
    w, v = _solve_interpolation(train_points, train_values, order, regularization_weight)
    query_values = _apply_interpolation(query_points, train_points, w, v, order)
    
    return query_values


def create_ndgrid(dims, normalize=True, center=False, dtype=torch.float32, device='cpu'):
    """
    Create n-dimensional grid coordiantes.
    Args:
        dims: Grid dimensions, e.g.[5, 5]
        normalize: whether to normalize to [0,1]
        center: whether to center at zero
        dtype: data type
    Returns:
        Grid coordinates [1, num_points, len(dims)]
    """
    if len(dims) == 2:
        # create 2D grid
        y, x = torch.meshgrid(torch.arange(dims[0]), torch.arange(dims[1]), indexing='ij')
        grid = torch.stack([y.flatten(), x.flatten()], dim=1).unsqueeze(0)  # [1, H*W, 2]
    elif len(dims) == 3:
        z, y, x = torch.meshgrid(torch.arange(dims[0]), torch.arange(dims[1]), torch.arange(dims[2]), indexing='ij')
        grid = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1).unsqueeze(0)  # [1, H*W*D, 3]
    else:
        raise ValueError(f"Unsupported dimension number: {len(dims)}")
    
    grid = grid.to(dtype=dtype, device=device)
    
    if normalize:
        grid = grid / (torch.tensor(dims, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0) - 1)
        if center:
            grid = (grid - 0.5) * 2
    
    return grid


class ThinPlateSplineSTN(nn.Module):
    def __init__(self, in_channels, resolution, control_points=(5, 5), order=2):
        super(ThinPlateSplineSTN, self).__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.control_points = control_points
        self.order = order
        
        output_params = control_points[0] * control_points[1] * 2  # 5*5*2=50
        
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels * 2, 20, kernel_size=5),  # the source and target anatomy features are concatenated 
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(20, 20, kernel_size=5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(20, 20, kernel_size=5),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate the size of the features after convolution
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, in_channels * 2, resolution, resolution)
        #     conv_output = self.localization(dummy_input)
        #     self.fc_input_size = conv_output.view(conv_output.size(0), -1).size(1)
        self.fc_input_size = (resolution // 4 -7)**2 * 20
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_input_size, 100),
            nn.Tanh(),
            nn.Linear(100, output_params)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()
        
        self.register_buffer('initial_cp_grid', None)
        self.register_buffer('target_grid', None)
        self._initialize_grids()
    
    def _initialize_grids(self):
        device = next(self.parameters()).device
        # create the initial control point grid and target image grid
        self.initial_cp_grid = create_ndgrid(
            self.control_points, 
            normalize=True, 
            center=False,
            device=device
        )
        self.target_grid = create_ndgrid(
            [self.resolution, self.resolution],
            normalize=True,
            center=False, 
            device=device
        )
    
    def forward(self, source_anatomy, target_anatomy):
        batch_size = source_anatomy.shape[0]
        device = source_anatomy.device
        if self.initial_cp_grid.device != device:
            self.initial_cp_grid = self.initial_cp_grid.to(device)
            self.target_grid = self.target_grid.to(device)
        
        # 1. Get control point offsets
        combined = torch.cat([source_anatomy, target_anatomy], dim=1)
        xs = self.localization(combined)
        xs = xs.view(xs.size(0), -1)
        control_point_offsets = self.fc_loc(xs)  # [B, 50]
        
        # 2. Reshape control point offsets [B, 25, 2]
        control_point_offsets = control_point_offsets.view(batch_size, -1, 2)
        
        # 3. Calculate deformed control points
        # Expand initial control point grid to batch size
        initial_cp_grid_batch = self.initial_cp_grid.expand(batch_size, -1, -1)
        warped_cp_grid = initial_cp_grid_batch + control_point_offsets
        
        # 4. Expand target grid to batch size
        target_grid_batch = self.target_grid.expand(batch_size, -1, -1)
        
        # 5. Use thin plate spline to calculate deformation field
        # Interpolate from control points to target grid
        interpolated_coords = interpolate_spline(
            train_points=initial_cp_grid_batch,      # [B, 25, 2] initial control points
            train_values=warped_cp_grid,             # [B, 25, 2] deformed control points
            query_points=target_grid_batch,          # [B, H*W, 2] target pixel positions
            order=self.order                        
        )
        
        # 6. Handle coordinate system conversion
        # Keras version uses coordinate inversion and scaling
        interpolated_coords = torch.flip(interpolated_coords, dims=[-1])  # 交换x,y坐标
        
        # Scale to pixel coordinates [0, resolution-1]
        interpolated_coords = interpolated_coords * (self.resolution - 1)
        
        # 7. Reshape to grid shape [B, H, W, 2]
        transformation_grid = interpolated_coords.view(
            batch_size, self.resolution, self.resolution, 2
        )
        
        # 8. Convert to format required by PyTorch grid_sample [-1, 1]
        # Convert from pixel coordinates [0, resolution-1] to [-1, 1]
        transformation_grid = 2.0 * transformation_grid / (self.resolution - 1) - 1.0
        transformed_anatomy = F.grid_sample(
            source_anatomy, 
            transformation_grid, 
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return transformed_anatomy, transformation_grid


class AnatomyFuser(nn.Module):
    def __init__(self,
                 in_channels,
                 resolution,
                 control_points=(5, 5),
                 order=2
    ):
        super(AnatomyFuser, self).__init__()
        self.spatial_transformer = ThinPlateSplineSTN(
            in_channels=in_channels,
            resolution=resolution,
            control_points=control_points,
            order=order
        )
        
    def forward(self, source_anatomy, target_anatomy):
        deformed_anatomy, _ = self.spatial_transformer(source_anatomy, target_anatomy)

        fused_anatomy = torch.max(deformed_anatomy, target_anatomy)

        return deformed_anatomy, fused_anatomy


class MultiModalAnatomyFuser(nn.Module):
    def __init__(self, anatomy_shape, config, num_modalities=2):
        super(MultiModalAnatomyFuser, self).__init__()
        
        self.num_modalities = num_modalities
        self.anatomy_fusers = nn.ModuleList([
            AnatomyFuser(anatomy_shape, config) for _ in range(num_modalities)
        ])
        
    def forward(self, anatomies):
        results = []
        
        for i, fuser in enumerate(self.anatomy_fusers):
            # Use anatomy i as source, others as targets
            source = anatomies[i]
            
            # For simplicity, use the next modality as target (circular)
            target_idx = (i + 1) % len(anatomies)
            target = anatomies[target_idx]
            
            deformed, fused = fuser(source, target)
            results.append((deformed, fused))
        
        return results



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = AnatomyFuser(in_channels=8, resolution=128, control_points=(5, 5), order=2).to(device)
    source_anatomy = torch.randn(2, 8, 128, 128).to(device)
    target_anatomy = torch.randn(2, 8, 128, 128).to(device)
    deformed_anatomy, fused_anatomy = model(source_anatomy, target_anatomy)
    print(deformed_anatomy.shape)
    print(fused_anatomy.shape)