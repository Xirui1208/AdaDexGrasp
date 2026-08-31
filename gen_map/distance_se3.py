import torch

ESP = torch.finfo().eps * 4


def geodesic_distance_between_R(R1: torch.Tensor, R2: torch.Tensor):
    R = torch.mm(R1.t(), R2)
    traces = torch.trace(R)  # [bs]
    theta = torch.clamp(0.5 * (traces - 1), -1 + ESP, 1 - ESP)
    dist = torch.acos(theta)
    return dist


def geodesic_distance_between_R_batch(R1, R2):
    R1_T = R1.transpose(1, 2)
    R = torch.einsum("bmn,bnk->bmk", R1_T, R2)
    diagonals = torch.diagonal(R, dim1=1, dim2=2)
    traces = torch.sum(diagonals, dim=1).unsqueeze(1)  # [bs]
    theta = torch.clamp(0.5 * (traces - 1), -1 + ESP, 1 - ESP)
    dist = torch.acos(theta)

    return dist


def double_geodesic_distance_between_poses(T1, T2, return_both=False):
    R_1, t_1 = T1[:3, :3], T1[:3, 3]
    R_2, t_2 = T2[:3, :3], T2[:3, 3]

    dist_R_square = geodesic_distance_between_R(R_1, R_2) ** 2
    dist_t_square = torch.sum((t_1 - t_2) ** 2)
    dist = torch.sqrt(dist_R_square.squeeze(-1) + dist_t_square)

    if return_both:
        return torch.sqrt(dist_t_square), torch.sqrt(dist_R_square)
    else:
        return dist


def double_geodesic_distance_between_poses_batch(T1, T2, return_both=False):
    R_1, t_1 = T1[:, :3, :3], T1[:, :3, 3]
    R_2, t_2 = T2[:, :3, :3], T2[:, :3, 3]

    dist_R_square = geodesic_distance_between_R_batch(R_1, R_2) ** 2
    dist_t_square = torch.sum((t_1 - t_2) ** 2, dim=1)
    dist = torch.sqrt(dist_R_square.squeeze(-1) + dist_t_square)  # [bs]

    if return_both:
        return torch.sqrt(dist_t_square), torch.sqrt(dist_R_square)
    else:
        return dist
