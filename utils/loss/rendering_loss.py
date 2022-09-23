import sys 
sys.path.append(".") 
import torch.nn as nn
import torch

from utils.renderer import phong_renderer
from utils.renderer import phongshading_cuda

class sh_rendering_loss(nn.Module):
    def __init__(self, level = 2, size = 512):
        super(sh_rendering_loss, self).__init__()
        self.size = size
        self.renderer = phong_renderer.torch_SH_renderer(level=level, normal_size = size)
        self.criterion_MSE = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

    def forward(self, sh_coeff, normal, albedo, ground_truth, mask = None):       
        if mask is None:
            render_result = self.renderer(normal, sh_coeff, albedo)
            loss = self.criterion_MSE(render_result, ground_truth) + \
                   self.criterion_L1(render_result, ground_truth)
        else:
            render_result = self.renderer(normal * mask, sh_coeff, albedo * mask)
            # loss = self.criterion_MSE(render_result * mask, ground_truth * mask) + \
            #        self.criterion_L1(render_result * mask, ground_truth * mask)
            loss = self.criterion_L1(render_result * mask, ground_truth * mask)
        return loss

    def show_render_img(self, sh_coeff, normal, albedo):
        render_result = self.renderer(normal, sh_coeff, albedo)
        return render_result

class sh_rendering_with_Phongspecular_loss(nn.Module):
    def __init__(self, level = 2, size = 512):
        super(sh_rendering_with_Phongspecular_loss, self).__init__()
        self.size = size
        self.renderer = phong_renderer.torch_SH_renderer(level=level, normal_size = size)
        self.Phong = phongshading_cuda.PhongShading()
        self.criterion_MSE = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

    def forward(self, sh_coeff, normal, albedo, ground_truth, mask = None):       
        if mask is None:
            sh_render_result = self.renderer(normal, sh_coeff, albedo)
            env = self.renderer.shcoeff2shimg(sh_coeff)
            _, specular = self.Phong.shading(env, normal.permute(0, 2, 3, 1))
            specular = specular[0, 3:6, ]
            render_result = sh_render_result + specular * 0.2
            # loss = self.criterion_MSE(render_result, ground_truth) + \
            #        self.criterion_L1(render_result, ground_truth)
            loss = self.criterion_L1(render_result, ground_truth)
        else:
            sh_render_result = self.renderer(normal * mask, sh_coeff, albedo * mask)
            env = self.renderer.shcoeff2shimg(sh_coeff)
            _, specular = self.Phong.shading(env.detach().clone(), normal.permute(0, 2, 3, 1).detach().clone())
            specular = torch.from_numpy(specular[0, 3:6, ].detach().cpu().numpy()).to('cuda') * mask         
            render_result = sh_render_result + specular * 0.2
            # loss = self.criterion_MSE(render_result * mask, ground_truth * mask) + \
            #        self.criterion_L1(render_result * mask, ground_truth * mask)
            loss = self.criterion_L1(render_result * mask, ground_truth * mask)
        return loss, specular

    def show_render_img(self, sh_coeff, normal, albedo):
        render_result = self.renderer(normal, sh_coeff, albedo)
        return render_result


# class sh_rendering_loss(nn.Module):
#     def __init__(self, level = 5):
#         super(sh_rendering_loss, self).__init__()
#         self.renderer = phong_renderer.torch_SH_renderer(level=level)
#         self.criterion = nn.MSELoss()

#     def forward(self, sh_coeff, normal, albedo, ground_truth, mask = None):
#         render_result = self.renderer(normal, sh_coeff, albedo)
#         if mask is None:
#             loss = self.criterion(render_result, ground_truth)
#         else:
#             loss = self.criterion(render_result * mask, ground_truth * mask)
#         return loss

#     def show_render_img(self, sh_coeff, normal, albedo):
#         render_result = self.renderer(normal, sh_coeff, albedo)
#         return render_result


