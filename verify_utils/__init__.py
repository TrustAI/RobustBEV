from math import pi
from .metric import CentreDistance, CentreDistance_det
from .doo import (BaselineDIRECT,
                  LowBoundedDIRECT_potential)
from .verification_definations import (FunctionalVerification, 
        Hue_Saturation_Brightness_Verificaton,
        Hue_Saturation_Verificaton,
        Contrast_Verificaton, 
        Brightness_Verification,
        Shift_Verification,
        Scale_Verification,
        Shift_Scale_Verification,
        Motion_Blur_Verification)

task_dict={
    "o-hsb":Hue_Saturation_Brightness_Verificaton,
    "o-hs":Hue_Saturation_Verificaton,
    "o-c":Contrast_Verificaton,
    "o-b":Brightness_Verification,
    "g-sft":Shift_Verification,
    "g-sc":Scale_Verification,
    "g-sftsc":Shift_Scale_Verification,
}

def get_task_bound(args, nb_camera=6):
    bound = []
    if args.perturb_type == 'optical':
        mark = 'o-'
        if args.hue != 0:
            bound.append([-pi*args.hue, pi*args.hue])
            mark += "h"
        if args.saturation != 0.:
            bound.append([1-args.saturation, 1+args.saturation])
            mark += "s"
        if args.bright != 0.:
           bound.append([-args.bright, args.bright]) 
           mark += "b"
        if args.contrast != 0:
            bound.append([1-args.contrast, 1+args.contrast])
            mark += "c"
        return bound*nb_camera, task_dict[mark]

    elif args.perturb_type == 'geometry':
        mark = 'g-'
        if args.shift != 0.:
            bound.append([-args.shift, args.shift])
            bound.append([-args.shift, args.shift])
            mark += "sft"
        if args.scale != 0.:
            bound.append([1-args.scale, 1+args.scale])
            bound.append([1-args.scale, 1+args.scale])
            mark += "sc"
        return bound*nb_camera, task_dict[mark]
    elif args.perturb_type == 'motion':
        assert args.kernal_size != 0.
        bound.append([-180,180]) # angle
        bound.append([-1.,1.]) # direction
        return bound*nb_camera, Motion_Blur_Verification
    else:
        raise NotImplementedError

