import numpy as np


def jitter(pcd, std=0.01, clip=0.05):
    num_points, num_features = pcd.shape  # pcd.shape == (N, 3)
    jittered_point = np.clip(std * np.random.randn(num_points, num_features), -clip, clip)
    jittered_point += pcd
    return jittered_point

def jitter_with_normV(pcd, std=0.01, clip=0.05):
    num_points, _ = pcd.shape  # pcd.shape == (N, 3)
    jittered_point = np.clip(std * np.random.randn(num_points, 3), -clip, clip)
    jittered_point += pcd[:, :3]
    jittered_point = np.concatenate((jittered_point, pcd[:, 3:]), axis=-1)
    return jittered_point


def rotate(pcd, which_axis, angle_range):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    angle = np.pi * angle / 180
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    if which_axis == 'x':
        rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]])
    elif which_axis == 'y':
        rotation_matrix = np.array([[cos_theta, 0,  -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]])
    elif which_axis == 'z':
        rotation_matrix = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
    else:
        raise ValueError(f'which_axis should be one of x, y and z, but got {which_axis}!')
    rotated_points = pcd @ rotation_matrix
    return rotated_points


def translate(pcd, x_range, y_range, z_range):
    num_points = pcd.shape[0]
    x_translation = np.random.uniform(x_range[0], x_range[1])
    y_translation = np.random.uniform(y_range[0], y_range[1])
    z_translation = np.random.uniform(z_range[0], z_range[1])
    x = np.full(num_points, x_translation)
    y = np.full(num_points, y_translation)
    z = np.full(num_points, z_translation)
    translation = np.stack([x, y, z], axis=-1)
    return pcd + translation


def translate_with_normV(pcd, x_range, y_range, z_range):
    num_points = pcd.shape[0]
    x_translation = np.random.uniform(x_range[0], x_range[1])
    y_translation = np.random.uniform(y_range[0], y_range[1])
    z_translation = np.random.uniform(z_range[0], z_range[1])
    x = np.full(num_points, x_translation)
    y = np.full(num_points, y_translation)
    z = np.full(num_points, z_translation)
    translation = np.stack([x, y, z], axis=-1)
    translated_pcd = np.concatenate((translation, pcd[:, 3:]), axis=-1)
    return translated_pcd

def anisotropic_scale(pcd, x_range, y_range, z_range):
    x_factor = np.random.uniform(x_range[0], x_range[1])
    y_factor = np.random.uniform(y_range[0], y_range[1])
    z_factor = np.random.uniform(z_range[0], z_range[1])
    scale_matrix = np.array([[x_factor, 0, 0], [0, y_factor, 0], [0, 0, z_factor]])
    scaled_points = pcd @ scale_matrix
    return scaled_points


def anisotropic_scale_with_normV(pcd, x_range, y_range, z_range):
    x_factor = np.random.uniform(x_range[0], x_range[1])
    y_factor = np.random.uniform(y_range[0], y_range[1])
    z_factor = np.random.uniform(z_range[0], z_range[1])
    scale_matrix = np.array([[x_factor, 0, 0], [0, y_factor, 0], [0, 0, z_factor]])
    scaled_points = pcd[:, :3] @ scale_matrix
    scaled_points = np.concatenate((scaled_points, pcd[:, 3:]), axis=-1)
    return scaled_points


def rotate_with_normV(data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx6 array, original batch of point clouds and point normals
        Return:
          Nx6 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Extract the point coordinates and normals
    shape_pc = data[:, 0:3]
    shape_normal = data[:, 3:6]

    # Apply the rotation to the point coordinates and normals
    rotated_data[:, 0:3] = np.dot(shape_pc, R)
    rotated_data[:, 3:6] = np.dot(shape_normal, R)

    return rotated_data
