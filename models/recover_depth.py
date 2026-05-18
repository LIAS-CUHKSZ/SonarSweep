import torch
def compute_depth_norm_from_plane(sonar_plane_dist, K_c, T_cs, n_s, alpha_rad):
    """
    Compute the Euclidean camera-depth norm from estimated sonar plane distances.

    Args:
        sonar_plane_dist (torch.Tensor): Estimated sonar plane distance d*(u, v),
            shaped [B, 1, H, W].
        K_c (torch.Tensor): Camera intrinsics, shaped [B, 3, 3].
        T_cs (torch.Tensor): Transform from camera to sonar coordinates, shaped
            [B, 4, 4].
        n_s (torch.Tensor): Plane normal in sonar coordinates, shaped [B, 3, 1].
        alpha_rad (float or torch.Tensor): Plane tilt angle in radians.

    Returns:
        torch.Tensor: Euclidean depth norm for each pixel, shaped [B, 1, H, W].
    """
    # Initial setup.
    device = sonar_plane_dist.device
    dtype = sonar_plane_dist.dtype
    B, _, H, W = sonar_plane_dist.shape

    # Extract R_cs and t_cs from the 4x4 transform.
    R_cs = T_cs[:, :3, :3]  # Shape: [B, 3, 3]
    t_cs = T_cs[:, :3, 3:4] # Shape: [B, 3, 1]

    # Compute camera ray directions and their norms.
    # This depends only on K_c and image size and can be cached for speed.
    u_coords = torch.arange(W, device=device, dtype=dtype)
    v_coords = torch.arange(H, device=device, dtype=dtype)
    K_c = K_c.to(device)
    T_cs = T_cs.to(device)
    n_s = n_s.to(device)
    if isinstance(alpha_rad, torch.Tensor):
        alpha_rad = alpha_rad.to(device)
    else:
        alpha_rad = torch.tensor(alpha_rad, dtype=dtype, device=device)
        
    v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
    ones = torch.ones_like(u_grid)
    
    pixel_coords = torch.stack([u_grid, v_grid, ones], dim=0)       # [3, H, W]
    pixel_coords = pixel_coords.view(3, H * W)                       # [3, H*W]

    K_c_inv = torch.inverse(K_c)                                     # [B, 3, 3]
    cam_ray_dir = K_c_inv @ pixel_coords                             # [B, 3, H*W]
    
    # Ray-direction norm, used as the geometric scale factor.
    ray_norm = torch.linalg.norm(cam_ray_dir, ord=2, dim=1, keepdim=True) # [B, 1, H*W]
    
    # Compute camera depth Z_c.
    plane_normal_cam = R_cs @ n_s                                    # [B, 3, 1]
    plane_offset = plane_normal_cam.transpose(1, 2) @ t_cs           # [B, 1, 1]
    
    sin_alpha = torch.sin(alpha_rad)
    
    # Reshape d* to match H*W.
    sonar_plane_dist_flat = sonar_plane_dist.view(B, 1, H * W)       # [B, 1, H*W]
    numerator_Zc = sonar_plane_dist_flat * sin_alpha + plane_offset  # [B, 1, H*W]

    # Denominator: dot product n_c^T * cam_ray_dir.
    denominator_Zc = plane_normal_cam.transpose(1, 2) @ cam_ray_dir  # [B, 1, H*W]
    
    epsilon = 1e-8
    Z_c_flat = numerator_Zc / (denominator_Zc + epsilon)             # [B, 1, H*W]

    # Compute the final depth norm.
    # depth_norm = Z_c * norm(cam_ray_dir)
    depth_norm_flat = Z_c_flat * ray_norm                            # [B, 1, H*W]

    # Restore the image shape [B, 1, H, W].
    return depth_norm_flat.view(B, 1, H, W)
