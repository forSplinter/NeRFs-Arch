import torch

class CameraIntrinsics:
    def __init__(self, fl_x, fl_y, k1, k2, p1, p2, cx, cy,w, h, camera_angle_x=None, camera_angle_y=None, aabb_scale=16, is_fisheye=False, device='cpu'):
        """_summary_

        Args:
            fl_x (_type_): _description_
            fl_y (_type_): _description_
            k1 (_type_): _description_
            k2 (_type_): _description_
            p1 (_type_): _description_
            p2 (_type_): _description_
            cx (_type_): _description_
            w (_type_): _description_
            h (_type_): _description_
            camera_angle_x (_type_, optional): _description_. Defaults to None.
            camera_angle_y (_type_, optional): _description_. Defaults to None.
            aabb_scale (int, optional): _description_. Defaults to 16.
            is_fisheye (bool, optional): _description_. Defaults to False.
        """
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.camera_angle_x = camera_angle_x
        self.camera_angle_y = camera_angle_y
        self.aabb_scale = aabb_scale
        self.is_fisheye = is_fisheye
        self.device = device
    
    def get_focal(self):
        """_summary_
        """
        return torch.tensor([self.fl_x, self.fl_y], dtype=torch.float32, device=self.device)
    
    def get_ppoint(self):
        """_summary_
        """
        return torch.tensor([self.cx, self.cy], dtype=torch.float32)
    
    def get_distortion(self):
        """_summary_
        """
        return torch.tensor([self.k1, self.k2, self.p1, self.p2], dtype=torch.float32, device=self.device)
    
    def get_K(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        K = torch.tensor([
            [self.fl_x, 0.0, self.cx],
            [0.0, self.fl_y, self.cy],
            [0.0,       0.0,     1.0]
        ], dtype=torch.float32)
        return K
    
    def undistort_points(self, points_2d):
        """_summary_

        Args:
            points_2d (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not self.is_fisheye:
            #TODO Implement standard pinhole camera undistortion 
            pass
        else:
            #TODO Implement fisheye undistortion 
            pass
        return points_2d 
    
    def to(self, device):
        """_summary_

        Args:
            device (_type_): _description_
        """
        self.device = device
        
    