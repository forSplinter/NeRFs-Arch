import random
import torch

def get_color_from_ray_points(densities, colors, pads):
    """
    Estimate color from N sample points of a ray

    :param densities : (N), densities approximated of sampled points along a ray
    :param colors : (N,(R,G,B)), RGB colors approximated of sampled points along a ray
    :param pads : (N), distance between sampled point i => i+1, for last point it's distance to end of scene
    """
    transmittances = get_transmittances(densities, pads)
    opacities = 1 - torch.exp(-densities*pads)
    color_pixel = (transmittances.unsqueeze(-1) * opacities.unsqueeze(-1) * colors).sum(dim=0)
    
    return color_pixel
    
def get_transmittances(densities, pads):
    """
    Compute transmittances for each points of a ray
    
    :param densities: (N)
    :param pads: (N)
    """
    product = densities * pads
    cumul_product = torch.cumsum(product, dim=0)
    cumul_product = torch.cat([torch.tensor([0.0]),cumul_product[:-1]], dim=0)
    transmittances = torch.exp(-cumul_product)

    return transmittances

def loss_ray(gd_color, estimated_color):
    """
    Compute classic NeRF loss for minibatch of N rays
    
    :param gd_color: (N,3) Ground truth color for each rays
    :param estimated_color: (N,3) Estimated color for each rays
    """
    return (estimated_color - gd_color).pow(2).sum()

def sample_rays(dataset, ray_batch_size):
    """
    Compute R rays from dataset of N images
    
    :param dataset: (N, (pose, directions, colors, rays_origins)), rays data for N images
    :param ray_batch_size: R:int, number of unique rays to compute
    
    return batch_rayons: dict(origins, directions, colors, image_index), dict of rays computed
    """

    batch_rayons = {
        "origins": [],
        "directions": [],
        "colors": [],
        "image_index": [],
    }

    # Select random image for each rays
    image_index = torch.randint(0, len(dataset)-1,(ray_batch_size,))
    batch_rayons["image_index"] = image_index
    
    for i in image_index:
        image_data = dataset[i.item()]
        
        # Select random pixel from image, select ray
        H,W = image_data["directions"].shape[:2]
        pixel_index = random.randint(0, H*W-1)

        # Retrieve ray origin, direction, color
        origin = image_data["rays_origins"].view(-1,3)[pixel_index]
        direction = image_data["directions"].view(-1,3)[pixel_index]
        color = image_data["colors"].view(-1,3)[pixel_index]

        batch_rayons["origins"].append(origin)
        batch_rayons["directions"].append(direction)
        batch_rayons["colors"].append(color)

    return batch_rayons