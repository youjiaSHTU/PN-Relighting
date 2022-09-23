import cv2
import os
import torch
from math import *
import numpy as np
from glob import glob
from tqdm import tqdm


class PhongShading:
    def __init__(self, batch_size=1, Normal_map_shape=(512, 512), HDR_map_shape=(16, 32, 3), device='cuda'):
        self.Normal_map_shape = Normal_map_shape
        self.HDR_map_shape = HDR_map_shape
        self.batch_size = batch_size
        self.device = device

        size = (batch_size, 3, Normal_map_shape[0], Normal_map_shape[1])
        self.lightmap_d = torch.zeros(size).type(torch.FloatTensor).to(device)
        self.lightmap_s = torch.zeros(size).type(torch.FloatTensor).to(device)
        self.lightmap_s16 = torch.zeros(size).type(
            torch.FloatTensor).to(device)
        self.lightmap_s32 = torch.zeros(size).type(
            torch.FloatTensor).to(device)
        self.lightmap_s64 = torch.zeros(size).type(
            torch.FloatTensor).to(device)

        self.Matl = torch.zeros(
            (batch_size, 3, HDR_map_shape[0] * HDR_map_shape[1])).type(torch.FloatTensor).to(device)
        self.Math = torch.zeros(
            (batch_size, 3, HDR_map_shape[0] * HDR_map_shape[1])).type(torch.FloatTensor).to(device)
        self.coeff = torch.zeros((batch_size, HDR_map_shape[0], 1, 3)).type(
            torch.FloatTensor).to(device)
        vec_v = torch.Tensor([0, 0, 1]).type(torch.FloatTensor).to(self.device)
        matl = torch.zeros(HDR_map_shape).type(torch.FloatTensor).to(device)
        math = torch.zeros(HDR_map_shape).type(torch.FloatTensor).to(device)
        for i in range(HDR_map_shape[0]):
            for j in range(HDR_map_shape[1]):
                phi = i/HDR_map_shape[0]*np.pi
                theta = j/HDR_map_shape[1]*np.pi*2
                matl[i][j] = torch.Tensor(
                    [np.sin(theta)*np.sin(phi), np.cos(phi), -np.cos(theta)*np.sin(phi)]).to(self.device)
                matl[i][j] /= torch.norm(matl[i][j])
                math[i][j] = (vec_v+matl[i][j])/torch.norm(vec_v+matl[i][j])
        matl = torch.t(torch.reshape(matl, (-1, 3)))
        math = torch.t(torch.reshape(math, (-1, 3)))
        for i in range(HDR_map_shape[0]):
            phi = i/HDR_map_shape[0]*np.pi
            self.coeff[0][i][0] = torch.Tensor([np.sin(phi), np.sin(
                phi), np.sin(phi)]).type(torch.FloatTensor).to(self.device)
        for i in range(batch_size):
            self.Matl[i, ] = matl
            self.Math[i, ] = math
            self.coeff[i, ] = self.coeff[0, ]

    def shading(self, hdr_map, Normal_map, split=0):
        # normal shape = [bs, H, W, 3]
        HDR_map = hdr_map * self.coeff
        Normal_map = torch.flip(Normal_map, [3])
        Normal_map = (Normal_map - 0.5) * 2
        Normal_map = torch.nn.functional.normalize(Normal_map, p=2, dim=3)
        self.Matn = torch.reshape(Normal_map, (self.batch_size, -1, 3))

        for channel in range(3):
            Matd = torch.bmm(self.Matn, self.Matl)
            Mats = torch.bmm(self.Matn, self.Math)
            Matd[Matd < 0] = 0
            Mats[Mats < 0] = 0

            Mats8 = (Mats / Mats.max()) ** 8
            Mats16 = Mats8 * Mats8
            Mats32 = Mats16 * Mats16
            Mats64 = Mats32 * Mats32

            Mat_hdr = torch.reshape(HDR_map[:, :, :, channel], (self.batch_size, 1, -1))
            Matd[:, :, :] *= Mat_hdr
            Mats[:, :, :] *= Mat_hdr
            Mats16[:, :, :] *= Mat_hdr
            Mats32[:, :, :] *= Mat_hdr
            Mats64[:, :, :] *= Mat_hdr
            size = (self.batch_size,
                    self.Normal_map_shape[0], self.Normal_map_shape[1])
            self.lightmap_d[:, channel, :, :] = torch.reshape(torch.sum(Matd, dim=2), size)
            self.lightmap_s[:, channel, :, :] = torch.reshape(torch.sum(Mats, dim=2), size)
            self.lightmap_s16[:, channel, :, :] = torch.reshape(torch.sum(Mats16, dim=2), size)
            self.lightmap_s32[:, channel, :, :] = torch.reshape(torch.sum(Mats32, dim=2), size)
            self.lightmap_s64[:, channel, :, :] = torch.reshape(torch.sum(Mats64, dim=2), size)

        lightmap_d = self.lightmap_d / 60
        lightmap_s = self.lightmap_s / 200
        lightmap_s16 = self.lightmap_s16 / 50
        lightmap_s32 = self.lightmap_s32 / 40
        lightmap_s64 = self.lightmap_s64 / 20

        return lightmap_d, torch.cat([lightmap_s, lightmap_s16, lightmap_s32, lightmap_s64], dim=1)


def hdr_rot(hdr, angle_in_deg=0.0):
    if angle_in_deg == 0.0:
        return hdr
    new_hdr = np.zeros_like(hdr)
    p = angle_in_deg/360.0
    h, w, _ = new_hdr.shape
    p = int(p*w)
    print(p)
    new_hdr[:, :p] = hdr[:, -1-p:-1]
    new_hdr[:, -1-p:] = hdr[:, :p]
    return new_hdr


if __name__ == "__main__":
    A = PhongShading(Normal_map_shape=(512, 512))
    normal = cv2.imread('/data/009.png') / 255.0
    albedo = cv2.imread('/data/20576.png') / 255.0
    mask = 1.0 - cv2.imread('/data/hekai/ffhq/FFHQ/filtered_back_mask/20576.png') / 255.0
    # env = cv2.imread('/data/hekai/512_jit/label/0000000001.hdr', -1)
    env = cv2.resize(np.load('/data/00000.npy'), (32, 16))
    # env = np.load('/data/new_hdr.npy') / 5
    # print(env_high.mean(), env_high.max())
    # env_sh = np.load('/data/sh_env.npy')
    # env = (env_high + env_sh)
    env = torch.from_numpy(env).to('cuda').to(torch.float32)
    normal = torch.from_numpy(normal).to('cuda').to(torch.float32)
    diffuse, specular = A.shading(env[None], normal[None])

    diffuse = diffuse[0].cpu().numpy().transpose(1, 2, 0)
    specular = specular[0].cpu().numpy().transpose(1, 2, 0)
    img_src = cv2.resize(cv2.imread('/data/hekai/ffhq/FFHQ/FFHQ/20576.png'), (512, 512)) / 255.0
    img_cat = np.concatenate([img_src * mask, diffuse * albedo * mask * 1.2, specular[:, :, 0:3] * mask, specular[:, :, 3:6] * mask, specular[:, :, 6:9] * mask, specular[:, :, 9:12] * mask], axis = 1)

    cv2.imwrite('/data/cat_sh.png', img_cat * 255)



    # normals = sorted(glob('/data/hekai/comparison/set_taeser/normal/*'))
    # albedos = sorted(glob('/data/hekai/comparison/set_taeser/albedo/*'))
    # masks = sorted(glob('/data/hekai/comparison/set_taeser/mask/*'))
    # envs = sorted(glob('/data/hekai/comparison/set_taeser/hdr/*'))
    # out = '/data/hekai/comparison/set_taeser/phong'
    # os.makedirs(out, exist_ok=True)
    # for i in tqdm(range(len(normals))):
    #     normal = cv2.imread(normals[i]).astype(np.float32) / 255
    #     # albedo = cv2.imread("/data/hekai/512_jit/albedo_refine/011/00043.png").astype(np.float32) / 255
    #     # mask = cv2.imread("/data/hekai/512_jit/matte/011/043_matte.png").astype(np.float32) / 255
    #     albedo = cv2.imread(albedos[i]).astype(np.float32) / 255
    #     mask = cv2.imread(masks[i]).astype(np.float32) / 255
    #     mask = cv2.resize(mask,(512,512))
    #     # env = np.load('/data/hekai/ffhq/FFHQ/optim_env2/35053.npy')
    #     # print(env.shape)
    #     # print(env.mean())
    #     env = cv2.resize(cv2.imread(envs[i], -1), (32, 16))
    #     cur = np.zeros_like(env)
    #     # move =  11 #offset
        
    #     # cur[:,0:32-move,:] = env [:,move:32,:]
    #     # cur[:,32-move:32,:] = env [:,0:move,: ]
    #     # env = cur
    #     #  spruit_sunrise_4k.hdr  0
    #     # the_sky_is_on_fire_4k.hdr  1
    #     # color2  2
    #     # ztt 443 brunch 3
    #     env = torch.from_numpy(env).to('cuda').to(torch.float32)
    #     normal = torch.from_numpy(normal).to('cuda').to(torch.float32)
    #     diffuse, specular = A.shading(env[None], normal[None])

    #     diffuse = diffuse[0].cpu().numpy().transpose(1, 2, 0)
    #     specular = specular[0].cpu().numpy().transpose(1, 2, 0)
    #     # img_cat = np.concatenate([diffuse * albedo, specular[:, :, 0:3], specular[:, :, 3:6], specular[:, :, 6:9], specular[:, :, 9:12]], axis = 1)

    #     abosulute_mask = mask
    #     abosulute_mask[abosulute_mask<0.5] = 0
    #     # abosulute_mask[abosulute_mask>=0.5] = 1
    #     # + specular[:, :, 6:9] * 0.6 
    #     # + (1-abosulute_mask)*0.6
    #     normal = normal.detach().cpu().numpy()  #normal
        
    #     img_cat = np.concatenate([diffuse * albedo + 0.10 *specular[:, :, 3:6] + 0.10 * specular[:, :, 6:9]], axis = 1)
    #     # img_cat = np.concatenate([albedo], axis = 1)
        
    #     # img_cat = img_cat * abosulute_mask+ (1-abosulute_mask) *0.55  #albedo
    #     # img_cat = img_cat * abosulute_mask  + (1-abosulute_mask) *([1,0.5,0.5])  #normal
    #     # mask = np.sum(mask, axis = 2, keepdims = True) / 3
    #     # img_cat = np.concatenate([img_cat, mask], axis = 2)
    #     mask = np.sum(mask, axis = 2, keepdims=True)
    #     img_cat = np.concatenate([img_cat, mask], axis = 2)
    #     img_cat = img_cat ** (1 / 1.7)
    #     cv2.imwrite(os.path.join(out, str(i).zfill(5) + '.png'), img_cat * 255)
