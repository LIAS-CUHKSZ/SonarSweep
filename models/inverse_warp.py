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
    # 0. Read dimensions.
    batch_size, _, sonar_height, sonar_width = sonar_rect.shape
    sonar_shape = (sonar_height, sonar_width)
    
    height, width = target_shape[-2], target_shape[-1]
    target_shape = (height, width)

   
    # 1. Map camera pixels to sonar pixels in batch form.
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
    
    # 2. Normalize coordinates for F.grid_sample.
    map_x = (map_x / (sonar_width - 1)) * 2.0 - 1.0
    map_y = (map_y / (sonar_height - 1)) * 2.0 - 1.0

    # 3. Build the sampling grid.
    grid = torch.stack((map_x, map_y), dim=-1)
    grid = grid.to(sonar_rect.dtype) 
    # grid shape is [B, H_t, W_t, 2], which is what grid_sample expects.

    # 4. Perform differentiable sampling.
    warped_feat = F.grid_sample(
        sonar_rect, 
        grid, 
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
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
    # 0. Normalize inputs and determine batch size.
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

    B = K_cam.shape[0]
    device = K_cam.device
    dtype = torch.float32
    height, width = target_shape

    # 1. Generate or reuse the cached pixel-coordinate grid (u, v).
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

    # 2. Precompute batch-compatible constants.
    R_sonar_from_cam = T_sonar_from_cam[:, :3, :3]       # Shape: [B, 3, 3]
    t_sonar_from_cam = T_sonar_from_cam[:, :3, 3:4]     # Shape: [B, 3, 1]

    cos_alpha = torch.cos(alpha) # Shape: [B]
    sin_alpha = torch.sin(alpha) # Shape: [B]

    M = K_cam @ R_sonar_from_cam  # Shape: [B, 3, 3]
    C = K_cam @ t_sonar_from_cam  # Shape: [B, 3, 1]

    m1, m2, m3 = M[:, 0, :], M[:, 1, :], M[:, 2, :] # Shapes: [B, 3]
    c1, c2, c3 = C[:, 0], C[:, 1], C[:, 2]         # Shapes: [B, 1]

    # 3. Build the vectorized linear systems.
    # A_sys shape: [B, H*W, 3, 3], b_sys shape: [B, H*W, 3, 1].
    A_sys = torch.zeros((B, num_pixels, 3, 3), dtype=dtype, device=device)
    b_sys = torch.zeros((B, num_pixels, 3, 1), dtype=dtype, device=device)

    # Equation A: plane constraint.
    plane_eq_row = torch.stack([torch.zeros_like(alpha), cos_alpha, sin_alpha], dim=1) # Shape: [B, 3]
    A_sys[:, :, 0, :] = plane_eq_row[:, None, :] # Broadcast [B, 1, 3] to [B, H*W, 3]
    b_sys[:, :, 0, 0] = depth * sin_alpha # Broadcast [B] to [B, H*W]

    # Equation B: u projection with broadcasting.
    # u_flat[None, :, None]: [1, H*W, 1], m3[:, None, :]: [B, 1, 3] => [B, H*W, 3]
    A_sys[:, :, 1, :] = u_flat[None, :, None] * m3[:, None, :] - m1[:, None, :]
    b_sys[:, :, 1, :] = c1[:, None, :] - u_flat[None, :, None] * c3[:, None, :]

    # Equation C: v projection with broadcasting.
    A_sys[:, :, 2, :] = v_flat[None, :, None] * m3[:, None, :] - m2[:, None, :]
    b_sys[:, :, 2, :] = c2[:, None, :] - v_flat[None, :, None] * c3[:, None, :]
    
    # 4. Solve all linear systems in batch.
    # `solve` expects [..., N, N] and [..., N, K], so merge B and H*W.
    A_sys_reshaped = A_sys.view(B * num_pixels, 3, 3)
    b_sys_reshaped = b_sys.view(B * num_pixels, 3, 1)
    
    sonar_P_3d_flat = torch.linalg.solve(A_sys_reshaped, b_sys_reshaped) # Shape: [B*H*W, 3, 1]
    sonar_P_3d = sonar_P_3d_flat.view(B, num_pixels, 3) # Reshape back: [B, H*W, 3]
    
    # 5. Map 3D sonar coordinates to sonar-image pixel coordinates.
    X1, Y1, Z1 = sonar_P_3d[..., 0], sonar_P_3d[..., 1], sonar_P_3d[..., 2] # Shape: [B, H*W]
    
    Z1 = torch.clamp(Z1, min=1e-6)

    theta = torch.atan2(X1, Z1) # Shape: [B, H*W]
    d = Z1 / torch.cos(theta)   # Shape: [B, H*W]

    sonar_height, sonar_width = sonar_shape
    
    theta_prime_flat = (sonar_width / theta_range) * (torch.rad2deg(theta) + theta_range / 2)
    d_prime_flat = (sonar_height / distance_range) * d

    # 6. Reshape results back to image grids.
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

    # 1. Extract parameters and precompute constants.
    u, v = uv_coord
    
    # Extract extrinsic rotation R and translation t.
    R_sonar_from_cam = T_sonar_from_cam[:3, :3]
    t_sonar_from_cam = T_sonar_from_cam[:3, 3]


    # Precompute M and C.
    # M = K₂ * R₂₁ in our derivation
    M = K_cam @ R_sonar_from_cam 
    # C = K₂ * t₂₁ in our derivation
    C = K_cam @ t_sonar_from_cam

    m1, m2, m3 = M[0, :], M[1, :], M[2, :]
    c1, c2, c3 = C[0], C[1], C[2]

    # 2. Build the linear system A_sys * P1 = b_sys.
    A_sys = torch.zeros((3, 3), dtype=dtype, device=device)
    b_sys = torch.zeros((3, 1), dtype=dtype, device=device)

    # Equation A: plane constraint.
    A_sys[0, :] = torch.tensor([0, torch.cos(alpha_rad), torch.sin(alpha_rad)], dtype=dtype, device=device)
    b_sys[0] = depth * torch.sin(alpha_rad)

    # Equation B: u projection constraint.
    A_sys[1, :] = u * m3 - m1
    b_sys[1] = c1 - u * c3

    # Equation C: v projection constraint.
    A_sys[2, :] = v * m3 - m2
    b_sys[2] = c2 - v * c3

    # 3. Solve for P1, the 3D point in sonar coordinates.
    # P₁ = A_sys⁻¹ * b_sys, using a stable solver
    sonar_P_3d = torch.linalg.solve(A_sys, b_sys).squeeze()
    
    
    # 4. Map P1 to sonar-image pixel coordinates (d', theta').
    X1, Y1, Z1 = sonar_P_3d[0], sonar_P_3d[1], sonar_P_3d[2]
    
    # The angle is usually computed from the X-Z plane projection.
    theta = torch.atan2(X1, Z1)
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
    
    # 1. Define test parameters.
    target_width = 100
    target_height = int(np.ceil(100 / np.sqrt(3)))
    target_shape = (target_height, target_width)

    sonar_height, sonar_width = 150, 90
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cx = (target_width - 1) / 2.0
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

    # 2. Compute the full grid with the vectorized function.
    print("\nCalculating entire grid with vectorized function...")
    
    scale_factor = 4
    K_cam_scaled = K_cam.clone()
    K_cam_scaled[:2, :] /= scale_factor  # Scale down the focal lengths
    target_shape = (int(target_height/scale_factor), int(target_width/scale_factor))
    
    vectorized_d_grid, vectorized_theta_grid = transform_cam_grid_to_sonar_coords(
        target_shape, K_cam_scaled.squeeze(0), T_c_s.squeeze(0), depth, distance_range, theta_range, sonar_shape, alpha
    )
    print("Calculation complete.")

    # 3. Compare selected test pixels.
    all_tests_passed = True
    for i, uv in enumerate(test_pixel_uv):
        u_float, v_float = uv
        # Convert floating-point coordinates to integer indices for grid lookup.
        u_idx, v_idx = uv
        
        u_idx, v_idx = int(u_idx/scale_factor), int(v_idx/scale_factor)

        print(f"\n--- Test Case {i+1}: Camera Pixel (u={u_float:.2f}, v={v_float:.2f}) ---")
        
        # Compute the per-pixel reference result.
        expected_d, expected_theta = transform_cam_pixel_to_sonar_coords(
            (u_float, v_float), K_cam, T_c_s, depth, distance_range, theta_range, sonar_shape, alpha
        )
        print(f"  > Expected (per-pixel): (d'={expected_d:.4f}, θ'={expected_theta:.4f})")
        
        # Extract the matching pixel from the vectorized result.
        actual_d = vectorized_d_grid[:,v_idx, u_idx].item()
        actual_theta = vectorized_theta_grid[:, v_idx, u_idx].item()
        print(f"  > Actual   (vectorized): (d'={actual_d:.4f}, θ'={actual_theta:.4f})")
        
        # Compare the two results.
        try:
            torch.testing.assert_close(torch.tensor(actual_d), torch.tensor(expected_d), rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(torch.tensor(actual_theta), torch.tensor(expected_theta), rtol=1e-4, atol=1e-5)
            print("  PASSED")
        except AssertionError as e:
            print(f"  FAILED: Results do not match within tolerance.")
            print(f"     Difference in d': {abs(actual_d - expected_d)}")
            print(f"     Difference in θ': {abs(actual_theta - expected_theta)}")
            all_tests_passed = False

    print("\n--- Verification Summary ---")
    if all_tests_passed:
        print("All test cases passed. The vectorized implementation is correct.")
    else:
        print("Some test cases failed. Please review the vectorized implementation.")


if __name__ == '__main__':

    # test_two_coordinate_transform()

    test_vectorized_implementation()
