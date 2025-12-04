import torch 


class CameraExtrinsics:
    def __init__(self, transform_matrix):
        """_summary_

        Args:
            transform_matrix (_type_): _description_
        """
        self.c2w = torch.tensor(transform_matrix, dtype=torch.float32) # transformation from camera to world
        self.R = self.c2w[:3, :3] # Rotation matrix
        self.t = self.c2w[:3, 3].unsqueeze(1) # Translation vector
        
        self.w2c = self._compute_w2c()
    
    def _compute_w2c(self):
        """_summary_
        """
        R_inv = self.R.t() # Inverse of rotation matrix
        t_inv = -R_inv @ self.t
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = R_inv
        w2c[:3, 3:4] = t_inv

        return w2c
    
    def get_c2w(self):
        """_summary_
        """
        return self.c2w

    def get_w2c(self):
        """_summary_
        """
        return self.w2c
        
    def get_position(self):
        """_summary_
        """
        return self.t
    
    def get_up(self):
        """_summary_
        """
        return self.R[:,1]
    
    def get_right(self):
        """_summary_
        """
        return self.R[:,0]
    
    def world2camera(self, points_world):
        """_summary_

        Args:
            points_world (_type_): _description_

        Returns:
            _type_: _description_
        """
        if points_world.dim() == 1:
            points_world = points_world.unsqueeze(0) #make it (1,3)
        
        ones = torch.ones((points_world.shape[0], 1), dtype=torch.float32)
        pw_h = torch.cat([points_world, ones], dim=-1)  # (N,4)
        pc_h = (self.w2c @ pw_h.T).T  # (N,4)

        return pc_h[:, :3]
    
    def camera2world(self, points_camera):
        """_summary_

        Args:
            points_camera (_type_): _description_

        Returns:
            _type_: _description_
        """
        if points_camera.dim() == 1:
            points_camera = points_camera.unsqueeze(0) #make it (1,3)
        
        ones = torch.ones((points_camera.shape[0], 1), dtype=torch.float32)
        pc_h = torch.cat([points_camera, ones], dim=-1)  # (N,4)
        pw_h = (self.c2w @ pc_h.T).T  # (N,4)

        return pw_h[:, :3]