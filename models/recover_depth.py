import torch
import numpy as np
def compute_depth_norm_from_plane(sonar_plane_dist, K_c, T_cs, n_s, alpha_rad):
    """
    根据估算的声纳平面距离 d*，计算每个像素对应三维点 P_c 的欧几里得范数（深度）。
    这是一个无状态的、可微的函数。

    参数:
    - sonar_plane_dist (torch.Tensor): 估算的每个像素的声纳平面距离 d*(u,v)。形状 [B, 1, H, W]。
    - K_c (torch.Tensor): 相机内参矩阵。形状 [B, 3, 3]。
    - T_cs (torch.Tensor): 从相机坐标系到声纳坐标系的变换矩阵。形状 [B, 4, 4]。  T_sonar_from_cam
    - n_s (torch.Tensor): 声纳平面在声纳坐标系下的法向量。形状 [B, 3, 1]。
    - alpha_rad (float or torch.Tensor): 声纳平面的倾斜角（弧度）。

    返回:
    - depth_norm (torch.Tensor): 每个像素对应三维点的欧几里得范数（深度）。形状 [B, 1, H, W]。
    """
    # --- 初始设置 ---
    device = sonar_plane_dist.device
    dtype = sonar_plane_dist.dtype
    B, _, H, W = sonar_plane_dist.shape

    # --- 从 4x4 变换矩阵 T_cs 中提取 R_cs 和 t_cs ---
    R_cs = T_cs[:, :3, :3]  # 形状: [B, 3, 3]
    t_cs = T_cs[:, :3, 3:4] # 形状: [B, 3, 1]

    # --- 计算相机射线方向向量 cam_ray_dir 及其范数 ray_norm ---
    # 这一部分仅依赖于 K_c 和图像尺寸，可以被缓存以优化性能
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
    # 使用 @ 符号进行矩阵乘法，更现代且可读性好
    cam_ray_dir = K_c_inv @ pixel_coords                             # [B, 3, H*W]
    
    # 计算射线方向向量的范数，这是几何缩放因子
    ray_norm = torch.linalg.norm(cam_ray_dir, ord=2, dim=1, keepdim=True) # [B, 1, H*W]
    
    # --- 计算相机深度 Z_c ---
    plane_normal_cam = R_cs @ n_s                                    # [B, 3, 1]
    plane_offset = plane_normal_cam.transpose(1, 2) @ t_cs           # [B, 1, 1]
    
    sin_alpha = torch.sin(alpha_rad)
    
    # 将分子中的 d* 调整形状以匹配 H*W
    sonar_plane_dist_flat = sonar_plane_dist.view(B, 1, H * W)       # [B, 1, H*W]
    numerator_Zc = sonar_plane_dist_flat * sin_alpha + plane_offset  # [B, 1, H*W]

    # 分母: 点积 n_c^T * cam_ray_dir
    denominator_Zc = plane_normal_cam.transpose(1, 2) @ cam_ray_dir  # [B, 1, H*W]
    
    epsilon = 1e-8
    Z_c_flat = numerator_Zc / (denominator_Zc + epsilon)             # [B, 1, H*W]

    # --- 计算最终的深度范数 ---
    # depth_norm = Z_c * norm(cam_ray_dir)
    depth_norm_flat = Z_c_flat * ray_norm                            # [B, 1, H*W]

    # 将最终结果恢复为图像形状 [B, 1, H, W]
    return depth_norm_flat.view(B, 1, H, W)

# --- 使用示例 ---
def test_depth_computation():
    """
    根据指定参数测试 compute_depth_norm_from_plane 函数，并打印特定像素点的深度值。
    """
    print("--- 开始测试深度计算函数 ---")
    
    # --- 1. 设置测试参数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    
    B, H, W = 2, 116, 200
    
    # 构建相机内参矩阵 K_cam
    cx = (W - 1) / 2.0  # 319.5
    cy = (H - 1) / 2.0  # 239.5
    fx = cx
    fy = cx
    
    K_cam = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0,  1]
    ], dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
    
    # 模拟输入
    # 使用固定种子以保证每次测试结果一致
    torch.manual_seed(42)
    # d_star_output = torch.rand(B, 1, H, W, dtype=torch.float32) * 5 + 1 # 1-6米
    d_star_output = torch.full((B, 1, H, W), 10.0, dtype=torch.float32, device=device)

    # T_cs_mat = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    T_cs_mat = torch.tensor([[1,0,0,1],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]], dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
    
    # --- 1. 定义 alpha ---
    alpha_rad = torch.tensor(np.deg2rad(60), device=device)
 
    # 构建 n_s 向量
    n_s_vec = torch.tensor([
        [0.0],
        [torch.cos(alpha_rad)],
        [torch.sin(alpha_rad)]
    ], dtype=torch.float32, device=device)
    # ], dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
    # --- 3. 扩展 n_s_vec 到 batch 维度并移动到设备 ---
    
    # 将所有张量移动到目标设备
    K_cam = K_cam.to(device)
    d_star_output = d_star_output.to(device)
    T_cs_mat = T_cs_mat.to(device)
    n_s_vec = n_s_vec.to(device)

    print(f"输入参数形状: B={B}, H={H}, W={W}")
    
    # --- 2. 调用函数 ---
    depth_map = compute_depth_norm_from_plane(
        d_star_output, K_cam, T_cs_mat, n_s_vec, alpha_rad
    )
    print(f"输出 depth_norm 形状: {depth_map.shape}")
    
    # --- 3. 提取并打印特定像素的深度值 ---
    test_pixel_uv = [
        [0, 0], [W/2, 0], [W - 1, 0], 
        [0, H/2], [W/2, H/2], [W - 1, H/2],  # 注意：(W, H/2)会越界，已修正为(W-1, H/2)
        [0, H - 1], [W/2, H - 1], [W - 1, H - 1]
    ]
    
    print("\n--- 特定像素点的深度值 (仅显示 Batch 0) ---")
    for u, v in test_pixel_uv:
        # 确保坐标是整数以便索引
        u_idx, v_idx = int(u), int(v)
        
        # 从 batch 0 中提取深度值
        value = depth_map[0, 0, v_idx, u_idx].item()
        
        # 提取对应的输入 d* 值以供对比
        d_star_val = d_star_output[0, 0, v_idx, u_idx].item()
        
        print(f"  - 像素 ({u_idx:4d}, {v_idx:4d}): 输入 d*={d_star_val:.4f}, 输出深度={value:.4f} 米")
        
    print("\n--- 测试完成 ---")


if __name__ == '__main__':
    test_depth_computation()