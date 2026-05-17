from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
pixel_coords = None


def warp_camera_to_sonar(sonar_rect, K_cam, T_sonar_from_cam, target_shape,
                                 depth, distance_range, theta_range, alpha=torch.tensor(np.deg2rad(30))):
    """
    (Corrected & Batched Warp Function)
    Warps a batch of sonar feature maps to the camera view.

    Args:
        sonar_rect (torch.Tensor): Source sonar feature maps. [B, C, H_s, W_s]
        K_cam (torch.Tensor): Camera intrinsic matrices. [B, 3, 3]
        T_sonar_from_cam (torch.Tensor): Transformation matrices. [B, 4, 4]
        target_shape (tuple): The (height, width) of the target warped image.
        depth (torch.Tensor): Depth parameters. [B]
        distance_range (float): Sonar max distance range.
        theta_range (float): Sonar angular range.
        alpha (torch.Tensor): Plane angles in radians. [B]

    Returns:
        torch.Tensor: The warped feature map in the camera view. [B, C, H_t, W_t]
    """
    # --- 0. 获取维度信息 ---
    batch_size, _, sonar_height, sonar_width = sonar_rect.shape
    sonar_shape = (sonar_height, sonar_width)
    
    height, width = target_shape[-2], target_shape[-1]
    target_shape = (height, width)

   
    # --- 1. 计算从相机像素到声呐像素的映射关系 (批处理方式) ---
    # 调用支持批处理的函数，它会返回形状为 [B, H, W] 的映射
    map_y, map_x = transform_cam_grid_to_sonar_coords(
        target_shape=target_shape,
        K_cam=K_cam,
        T_sonar_from_cam=T_sonar_from_cam,
        depth=depth,
        distance_range=distance_range,
        theta_range=theta_range,
        sonar_shape=sonar_shape,
        alpha=alpha
    )
    
    # --- 2. 为 F.grid_sample 归一化坐标 ---
    # 所有操作现在都在批处理张量上进行
    map_x = (map_x / (sonar_width - 1)) * 2.0 - 1.0
    map_y = (map_y / (sonar_height - 1)) * 2.0 - 1.0

    # --- 3. 构建采样网格 ---
    # 因为 map_x 和 map_y 已经有了批次维度，所以 stack 的结果也自然会有
    grid = torch.stack((map_x, map_y), dim=-1)
    grid = grid.to(sonar_rect.dtype) 
    # grid 的形状现在是 [B, H_t, W_t, 2]，这正是 grid_sample 所需的！
    # 不再需要 .unsqueeze() 和 .expand()

    # --- 4. 执行可微分采样 ---
    warped_feat = F.grid_sample(
        sonar_rect, 
        grid, 
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True # 确保这与您的归一化逻辑匹配
    )

    return warped_feat


def transform_cam_grid_to_sonar_coords(target_shape, K_cam, T_sonar_from_cam, 
                                               depth, distance_range, theta_range, 
                                               sonar_shape, alpha):
    """
    (Optimized, Vectorized & Batched)
    Transforms grids of camera pixel coordinates to their corresponding sonar coordinates.

    Args:
        target_shape (tuple): The (height, width) of the target camera image.
        K_cam (torch.Tensor): Camera intrinsic matrix. Shape: [B, 3, 3] or [3, 3].
        T_sonar_from_cam (torch.Tensor): Transformation matrix. Shape: [B, 4, 4] or [4, 4].
        depth (torch.Tensor): Depth parameter of the plane. Shape: [B] or scalar.
        distance_range (float): Max distance range of the sonar (scalar, shared across batch).
        theta_range (float): Angular range (degrees) of the sonar (scalar, shared across batch).
        sonar_shape (tuple): (height, width) of the sonar image (shared across batch).
        alpha (torch.Tensor): Angle (radians) of the plane. Shape: [B] or scalar.

    Returns:
        tuple: (d_prime, theta_prime) coordinate grids of shape [B, H, W].
    """
    # --- 0. 输入标准化与批次大小确定 ---
    # 如果输入是单个矩阵/值，为其增加一个批次维度
    if K_cam.dim() == 2:
        K_cam = K_cam.unsqueeze(0)
    if T_sonar_from_cam.dim() == 2:
        T_sonar_from_cam = T_sonar_from_cam.unsqueeze(0)
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor([depth], device=K_cam.device, dtype=K_cam.dtype)
    if depth.dim() == 0:
        depth = depth.unsqueeze(0)
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor([alpha], device=K_cam.device, dtype=K_cam.dtype)
    if alpha.dim() == 0:
        alpha = alpha.unsqueeze(0)

    B = K_cam.shape[0] # 获取批次大小
    device = K_cam.device
    dtype = torch.float32
    height, width = target_shape

    # --- 1. 生成或获取缓存的像素坐标网格 (u, v) ---
    # 这部分与批处理无关，可以保持原样
    func = transform_cam_grid_to_sonar_coords
    if not hasattr(func, 'u_grid') or func.u_grid.shape != (height, width):
        # print("Generating and caching coordinate grid...")
        y_coords = torch.arange(height, device=device, dtype=dtype)
        x_coords = torch.arange(width, device=device, dtype=dtype)
        v_grid, u_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        func.u_grid = u_grid
        func.v_grid = v_grid

    u_flat = func.u_grid.flatten() # Shape: [H*W]
    v_flat = func.v_grid.flatten() # Shape: [H*W]
    num_pixels = u_flat.shape[0]

    # --- 2. 预计算常量 (批处理兼容) ---
    R_sonar_from_cam = T_sonar_from_cam[:, :3, :3]       # Shape: [B, 3, 3]
    t_sonar_from_cam = T_sonar_from_cam[:, :3, 3:4]     # Shape: [B, 3, 1]

    cos_alpha = torch.cos(alpha) # Shape: [B]
    sin_alpha = torch.sin(alpha) # Shape: [B]

    M = K_cam @ R_sonar_from_cam  # Shape: [B, 3, 3]
    C = K_cam @ t_sonar_from_cam  # Shape: [B, 3, 1]

    m1, m2, m3 = M[:, 0, :], M[:, 1, :], M[:, 2, :] # Shapes: [B, 3]
    c1, c2, c3 = C[:, 0], C[:, 1], C[:, 2]         # Shapes: [B, 1]

    # --- 3. 向量化构建线性系统 (批处理兼容) ---
    # A_sys 形状: [B, H*W, 3, 3], b_sys 形状: [B, H*W, 3, 1]
    A_sys = torch.zeros((B, num_pixels, 3, 3), dtype=dtype, device=device)
    b_sys = torch.zeros((B, num_pixels, 3, 1), dtype=dtype, device=device)

    # 方程 A (平面约束)
    plane_eq_row = torch.stack([torch.zeros_like(alpha), cos_alpha, sin_alpha], dim=1) # Shape: [B, 3]
    A_sys[:, :, 0, :] = plane_eq_row[:, None, :] # Broadcast [B, 1, 3] to [B, H*W, 3]
    b_sys[:, :, 0, 0] = depth * sin_alpha # Broadcast [B] to [B, H*W]

    # 方程 B (投影 u): 使用 broadcasting
    # u_flat[None, :, None]: [1, H*W, 1], m3[:, None, :]: [B, 1, 3] => [B, H*W, 3]
    A_sys[:, :, 1, :] = u_flat[None, :, None] * m3[:, None, :] - m1[:, None, :]
    b_sys[:, :, 1, :] = c1[:, None, :] - u_flat[None, :, None] * c3[:, None, :]

    # 方程 C (投影 v): 使用 broadcasting
    A_sys[:, :, 2, :] = v_flat[None, :, None] * m3[:, None, :] - m2[:, None, :]
    b_sys[:, :, 2, :] = c2[:, None, :] - v_flat[None, :, None] * c3[:, None, :]
    
    # --- 4. 批量求解所有线性系统 ---
    # `solve` 需要 [..., N, N] 和 [..., N, K] 形状，我们将 B 和 H*W 合并
    A_sys_reshaped = A_sys.view(B * num_pixels, 3, 3)
    b_sys_reshaped = b_sys.view(B * num_pixels, 3, 1)
    
    sonar_P_3d_flat = torch.linalg.solve(A_sys_reshaped, b_sys_reshaped) # Shape: [B*H*W, 3, 1]
    sonar_P_3d = sonar_P_3d_flat.view(B, num_pixels, 3) # Reshape back: [B, H*W, 3]
    
    # --- 5. 将三维坐标映射到声呐图像像素坐标 (批处理兼容) ---
    X1, Y1, Z1 = sonar_P_3d[..., 0], sonar_P_3d[..., 1], sonar_P_3d[..., 2] # Shape: [B, H*W]
    
    Z1 = torch.clamp(Z1, min=1e-6)

    theta = torch.atan2(X1, Z1) # Shape: [B, H*W]
    d = Z1 / torch.cos(theta)   # Shape: [B, H*W]

    sonar_height, sonar_width = sonar_shape
    
    theta_prime_flat = (sonar_width / theta_range) * (torch.rad2deg(theta) + theta_range / 2)
    d_prime_flat = (sonar_height / distance_range) * d

    # --- 6. 将结果重塑为图像网格形状 (批处理兼容) ---
    theta_prime = theta_prime_flat.view(B, height, width)
    d_prime = d_prime_flat.view(B, height, width)
    
    return d_prime, theta_prime


# transfrom pixel is for testing
def transform_cam_pixel_to_sonar_coords(uv_coord, K_cam, T_sonar_from_cam, 
                                                depth, distance_range, theta_range, 
                                                sonar_shape, alpha_rad):
    """
    Transforms a single pixel coordinate (u, v) from the camera image to its 
    corresponding coordinate (d', theta') in the sonar image using a linear system approach.

    Args:
        uv_coord (list or tuple): The (u, v) pixel coordinate in the camera image.
        K_cam (torch.Tensor): The 3x3 camera intrinsic matrix.
        T_sonar_from_cam (torch.Tensor): The 4x4 extrinsic transformation matrix that transforms
                                        a point from sonar coords (cam1) to camera coords (cam2).
        depth (float): The depth parameter defining the plane's position in sonar coordinates.
        distance_range (float): The maximum distance range of the sonar.
        theta_range (float): The angular range (in degrees) of the sonar.
        sonar_shape (tuple): The (height, width) of the sonar image.
        alpha (float): The angle 'theta' (in radius) that the plane makes with the z-axis.

    Returns:
        tuple: The transformed (d_prime, theta_prime) coordinates in the sonar image.
    """
    device = K_cam.device
    dtype = torch.float32

    # --- 1. 提取参数和预计算 ---
    u, v = uv_coord
    
    # 提取外参的旋转 R 和平移 t
    R_sonar_from_cam = T_sonar_from_cam[:3, :3]
    t_sonar_from_cam = T_sonar_from_cam[:3, 3]


    # 预计算 M 和 C 矩阵/向量
    # M = K₂ * R₂₁ in our derivation
    M = K_cam @ R_sonar_from_cam 
    # C = K₂ * t₂₁ in our derivation
    C = K_cam @ t_sonar_from_cam

    m1, m2, m3 = M[0, :], M[1, :], M[2, :]
    c1, c2, c3 = C[0], C[1], C[2]

    # --- 2. 构建线性系统 A_sys * P₁ = b_sys ---
    A_sys = torch.zeros((3, 3), dtype=dtype, device=device)
    b_sys = torch.zeros((3, 1), dtype=dtype, device=device)

    # 方程 A: 平面约束
    A_sys[0, :] = torch.tensor([0, torch.cos(alpha_rad), torch.sin(alpha_rad)], dtype=dtype, device=device)
    b_sys[0] = depth * torch.sin(alpha_rad)

    # 方程 B: 投影约束 (u)
    A_sys[1, :] = u * m3 - m1
    b_sys[1] = c1 - u * c3

    # 方程 C: 投影约束 (v)
    A_sys[2, :] = v * m3 - m2
    b_sys[2] = c2 - v * c3

    # --- 3. 求解 P₁, 即声呐坐标系下的三维点 ---
    # P₁ = A_sys⁻¹ * b_sys, using a stable solver
    sonar_P_3d = torch.linalg.solve(A_sys, b_sys).squeeze()
    
    
    # --- 4. 将三维坐标 (P₁) 映射到声呐图像的像素坐标 (d', θ') ---
    X1, Y1, Z1 = sonar_P_3d[0], sonar_P_3d[1], sonar_P_3d[2]
    
    # 角度通常是基于 X-Z 平面的投影
    theta = torch.atan2(X1, Z1) # 使用 atan2 更稳定，能处理所有象限
    d = Z1/torch.cos(theta)

    sonar_height, sonar_width = sonar_shape
    theta_prime = (sonar_width / theta_range) * (torch.rad2deg(theta) + theta_range / 2)
    d_prime = (sonar_height / distance_range) * d

    return d_prime.item(), theta_prime.item()


def test_vectorized_implementation():
    """
    Verifies that the vectorized `transform_cam_grid_to_sonar_coords` function 
    produces the same results as the per-pixel `transform_cam_pixel_to_sonar_coords`.
    """
    print("\n--- Running Verification for Vectorized Implementation ---")
    
    # --- 1. 定义测试参数 (与你的测试函数完全一致) ---
    target_width = 100
    target_height = int(np.ceil(100 / np.sqrt(3))) # 确保为整数
    target_shape = (target_height, target_width)

    sonar_height, sonar_width = 150, 90
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cx = (target_width - 1) / 2.0  # 使用 (W-1)/2.0 作为中心更标准
    cy = (target_height - 1) / 2.0
    fx = cx
    fy = cx

    K_cam = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0,  1]
    ], dtype=torch.float32, device=device)
    
    T_c_s = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    depth = torch.tensor(10.0, device=device)
    distance_range = 15.0
    theta_range = 90.0
    alpha = torch.tensor(np.deg2rad(60.0), device=device)
    sonar_shape = (sonar_height, sonar_width)
    
    test_pixel_uv = [[0, 0], [50, 0], [target_width - 1, 0], 
                     [0, int(50/np.sqrt(3))], [50, int(50/np.sqrt(3))], [target_width - 1, int(50/np.sqrt(3))],
                     [0, target_height - 1], [50, target_height - 1], [target_width - 1, target_height - 1]]

    # --- 2. 使用向量化函数计算全网格结果 ---
    print("\nCalculating entire grid with vectorized function...")
    
    # TODO
    scale_factor = 4
    K_cam_scaled = K_cam.clone()
    K_cam_scaled[:2, :] /= scale_factor  # Scale down the focal lengths
    target_shape = (int(target_height/scale_factor), int(target_width/scale_factor))
    
    vectorized_d_grid, vectorized_theta_grid = transform_cam_grid_to_sonar_coords(
        target_shape, K_cam_scaled.squeeze(0), T_c_s.squeeze(0), depth, distance_range, theta_range, sonar_shape, alpha
    )
    print("Calculation complete.")

    # --- 3. 循环遍历测试点，进行对比 ---
    all_tests_passed = True
    for i, uv in enumerate(test_pixel_uv):
        u_float, v_float = uv
        # 将浮点坐标转换为整数索引，用于从网格中提取数据
        u_idx, v_idx = uv
        
        # TODO
        u_idx, v_idx = int(u_idx/scale_factor), int(v_idx/scale_factor)

        print(f"\n--- Test Case {i+1}: Camera Pixel (u={u_float:.2f}, v={v_float:.2f}) ---")
        
        # 获取“黄金标准”结果
        expected_d, expected_theta = transform_cam_pixel_to_sonar_coords(
            (u_float, v_float), K_cam, T_c_s, depth, distance_range, theta_range, sonar_shape, alpha
        )
        print(f"  > Expected (per-pixel): (d'={expected_d:.4f}, θ'={expected_theta:.4f})")
        
        # 从向量化结果中提取对应像素的结果
        actual_d = vectorized_d_grid[:,v_idx, u_idx].item()
        actual_theta = vectorized_theta_grid[:, v_idx, u_idx].item()
        print(f"  > Actual   (vectorized): (d'={actual_d:.4f}, θ'={actual_theta:.4f})")
        
        # 比较结果
        try:
            # torch.testing.assert_close 提供了容差比较
            torch.testing.assert_close(torch.tensor(actual_d), torch.tensor(expected_d), rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(torch.tensor(actual_theta), torch.tensor(expected_theta), rtol=1e-4, atol=1e-5)
            print("  ✅ PASSED")
        except AssertionError as e:
            print(f"  ❌ FAILED: Results do not match within tolerance.")
            print(f"     Difference in d': {abs(actual_d - expected_d)}")
            print(f"     Difference in θ': {abs(actual_theta - expected_theta)}")
            all_tests_passed = False

    print("\n--- Verification Summary ---")
    if all_tests_passed:
        print("🎉 All test cases passed! The vectorized implementation is correct. 🎉")
    else:
        print("🔥 Some test cases failed. Please review the vectorized implementation. 🔥")


if __name__ == '__main__':

    # test_two_coordinate_transform() # 你可以取消注释来运行你的原始测试

    test_vectorized_implementation()
    