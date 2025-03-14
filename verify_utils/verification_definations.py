import numpy as np
import torch
from math import pi
import copy
from kornia.color import (rgb_to_hsv, hsv_to_rgb)
from kornia.filters import motion_blur
from .perturb_utils import (factor2tensor, AffineTransf)


class FunctionalVerification():
    """
    Base class for defining verification
    """

    def __init__(self, 
                 model, 
                 sample_ind:int,
                 frame:dict, 
                 ori_match:set,
                 dist_th:float,
                 overshoot:float,
                 verify_loss,
                 vov_bb=False,
                 kernal_size:int=5): 

        self.img_mean = torch.tensor([103.530, 116.280, 123.675]) 
        if vov_bb:
            self.img_std = torch.tensor([57.375, 57.120, 58.395])
        else:
            self.img_std = torch.tensor([1.,1.,1.])

        self.model = model
        self.ori_frame = frame
        self.sample_ind = sample_ind

        self.ori_images = self.get_img_tensor(frame)
        self.nb_camera = self.ori_images.shape[0]
        self.rgb_images, self.min_max_vaules = self.processing()

        self.loss = verify_loss
        self.dist_th = dist_th
        self.ot = overshoot
        self.ori_match = ori_match 

        # motion blur only
        self.kernal_size = kernal_size


    def processing(self):
        rgb_imgs = torch.zeros_like(self.ori_images)
        min_max_values = []
        for ii in range(self.nb_camera):
            cur_img, min_max_v = self._normlise_img(self.ori_images[ii])
            rgb_imgs[ii] = cur_img[[2,1,0],...]
            min_max_values.append(min_max_v)
        return rgb_imgs, min_max_values
    
    @staticmethod
    def _normlise_img(img_tensor:torch.Tensor) -> torch.Tensor:
        return ((img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()), (img_tensor.min(), img_tensor.max()))

    def _unnormlise_img(self, img_tensor:torch.Tensor, min_v, max_v)\
                     -> torch.Tensor:
        return ((img_tensor * (max_v - min_v) + min_v)[[1,2,0],...] 
                - self.img_mean[:,None,None]) / self.img_std[:,None,None]

    def get_img_tensor(self, mmcv_data:dict) -> torch.Tensor:
        return (mmcv_data['img'][0].data[0].squeeze() * self.img_std[:,None,None]) \
                + self.img_mean[:,None,None]

    @staticmethod
    def replace_img_tensor(ori_frame, perturb_imgs:torch.Tensor) -> dict:
        new_frame = copy.deepcopy(ori_frame)
        new_frame['img'][0].data[0].data = perturb_imgs.data
        return new_frame

    def functional_perturb(self):
        raise NotImplementedError

    def verification(self, in_arrs):
        query_result = []
        for idx, in_arr in enumerate(in_arrs):
            perturbed_imgs = torch.zeros_like(self.rgb_images)
            in_arr = in_arr.reshape(self.nb_camera,-1)
            for i in range(self.nb_camera):
                perturbed_imgs[i] = self.functional_perturb(self.rgb_images[i],
                                            in_arr[i], self.min_max_vaules[i])
            tmp_frame = self.replace_img_tensor(self.ori_frame, perturbed_imgs.unsqueeze(0))
            with torch.no_grad():
                _pediction = self.model(return_loss=False, rescale=True, **tmp_frame)
            _query = self.loss(_pediction, self.sample_ind, 
                               self.dist_th, self.ori_match, True, self.ot)
            query_result.append(_query)
        return np.array(query_result)

    def set_problem(self):
        return self.verification

class Hue_Saturation_Verificaton(FunctionalVerification):

    def functional_perturb(self, x, in_arr, min_max_v):
        transformed = torch.zeros_like(x)
        hue, saturation = in_arr[0], in_arr[1]
        hue = factor2tensor(hue, 
                        x.device, x.dtype, x.shape)
        saturation = factor2tensor(saturation, 
                            x.device, x.dtype, x.shape)

        with torch.no_grad():
            x = rgb_to_hsv(x)
           # unpack the hsv values
            h, s, v = torch.chunk(x, chunks=3, dim=-3) 
            # hue: transform the hue value and apply mod
            divisor: float = 2 * pi
            h_out = torch.fmod(h + hue, divisor)
            # saturation
            s_out: torch.Tensor = torch.clamp(s * saturation, min=0., max=1.)
            x: torch.Tensor = torch.cat([h_out, s_out, v], dim=-3)
            x = hsv_to_rgb(x)

        x = self._unnormlise_img(x, *min_max_v)
        transformed = x
        return transformed 

class Brightness_Verification(FunctionalVerification):

    def functional_perturb(self, x, in_arr, min_max_v):
        transformed = torch.zeros_like(x)
        bright = in_arr[0]
        bright = factor2tensor(bright, 
                            x.device, x.dtype, x.shape)
        with torch.no_grad():
            x = rgb_to_hsv(x)
            # unpack the hsv values
            h, s, v = torch.chunk(x, chunks=3, dim=-3)  
            v_out: torch.Tensor = torch.clamp(v + bright, min=0., max=1.)
            x: torch.Tensor = torch.cat([h, s, v_out], dim=-3)
            x = hsv_to_rgb(x)

        x = self._unnormlise_img(x, *min_max_v)
        transformed = x
        return transformed 

class Hue_Saturation_Brightness_Verificaton(FunctionalVerification):

    def functional_perturb(self, x, in_arr, min_max_v):
        transformed = torch.zeros_like(x)
        hue, saturation, bright = in_arr[0], in_arr[1], in_arr[2]
        hue = factor2tensor(hue, 
                        x.device, x.dtype, x.shape)
        saturation = factor2tensor(saturation, 
                            x.device, x.dtype, x.shape)
        bright = factor2tensor(bright, 
                            x.device, x.dtype, x.shape)

        with torch.no_grad():
            x = rgb_to_hsv(x)
            # unpack the hsv values
            h, s, v = torch.chunk(x, chunks=3, dim=-3)  
            # hue: transform the hue value and apply mod
            divisor: float = 2 * pi
            h_out = torch.fmod(h + hue, divisor)
            # saturation
            s_out: torch.Tensor = torch.clamp(s * saturation, min=0., max=1.)
            # brightness
            v_out: torch.Tensor = torch.clamp(v + bright, min=0., max=1.)
            x: torch.Tensor = torch.cat([h_out, s_out, v_out], dim=-3)
            x = hsv_to_rgb(x)

        x = self._unnormlise_img(x, *min_max_v)
        transformed = x
        return transformed 

class Contrast_Verificaton(FunctionalVerification):

    def functional_perturb(self, x, in_arr, min_max_v):
        transformed = torch.zeros_like(x)
        contrast = in_arr[0]
        contrast = factor2tensor(contrast, 
                        x.device, x.dtype, x.shape) 
        with torch.no_grad():
            x = torch.clamp(x * contrast, min=0., max=1.)
        x = self._unnormlise_img(x, *min_max_v)
        transformed = x
        return transformed 

class Shift_Verification(FunctionalVerification):

    def functional_perturb(self, x, in_arr, min_max_v):
        transformed = torch.zeros_like(x)
        x_shift, y_shift = in_arr[0], in_arr[1]
        cur_theta = torch.tensor([1,0,x_shift,
                                0,1,y_shift], 
                                dtype=torch.float32)
        with torch.no_grad():
            cur_aff = AffineTransf(cur_theta)
            x = cur_aff(x.unsqueeze(0))
        x = self._unnormlise_img(x.squeeze_(), *min_max_v)
        transformed = x
        return transformed 

class Scale_Verification(FunctionalVerification):

    def functional_perturb(self, x, in_arr, min_max_v):
        transformed = torch.zeros_like(x)
        x_scale, y_scale = in_arr[0], in_arr[1]
        cur_theta = torch.tensor([x_scale.item(),0,0,
                                0,y_scale.item(),0], dtype=torch.float32)
        with torch.no_grad():
            cur_aff = AffineTransf(cur_theta)
            x = cur_aff(x.unsqueeze(0))
        x = self._unnormlise_img(x.squeeze_(), *min_max_v)
        transformed = x
        return transformed 

class Shift_Scale_Verification(FunctionalVerification):

    def functional_perturb(self, x, in_arr, min_max_v):
        transformed = torch.zeros_like(x)
        x_shift, y_shift, x_scale, y_scale = \
            in_arr[0], in_arr[1], in_arr[2], in_arr[3]
        cur_theta = torch.tensor([x_scale,0,x_shift,
                                0,y_scale,y_shift], dtype=torch.float32)
        with torch.no_grad():
            cur_aff = AffineTransf(cur_theta)
            x = cur_aff(x.unsqueeze(0))
        x = self._unnormlise_img(x.squeeze_(), *min_max_v)
        transformed = x
        return transformed 

class Motion_Blur_Verification(FunctionalVerification):

    def verification(self, in_arrs):
        query_result = []
        for in_arr in in_arrs:
            in_arr = in_arr.reshape(self.nb_camera,-1)
            perturbed_imgs = self.functional_perturb(self.rgb_images, in_arr)
            tmp_frame = self.replace_img_tensor(self.ori_frame, perturbed_imgs.unsqueeze(0))
            with torch.no_grad():
                _pediction = self.model(return_loss=False, rescale=True, **tmp_frame)
            _query = self.loss(_pediction, self.sample_ind, 
                               self.dist_th, self.ori_match, True, self.ot)
            query_result.append(_query)
        return np.array(query_result)

    def functional_perturb(self, x, in_arr):
        transformed = torch.zeros_like(x)
        angle= torch.tensor(in_arr[:,0],dtype=torch.float32) #+ self.position
        direction = torch.tensor(in_arr[:,1],dtype=torch.float32) #+ self.position

        perturbed_imgs = motion_blur(x, self.kernal_size,
                                angle=angle, direction=direction,
                                mode='bilinear')
        for idx, img in enumerate(perturbed_imgs):
            transformed[idx] = self._unnormlise_img(img.squeeze_(), *self.min_max_vaules[idx])
        return transformed 