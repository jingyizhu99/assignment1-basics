from math import cos, pi

def learning_rate_schedule(t, lr_max, lr_min, Tw, Tc):
    lr = 0
    if t < Tw:
        lr = t/Tw * lr_max
    elif t > Tc:
        lr = lr_min
    else:
        cos_inner = pi * (t - Tw) / (Tc - Tw)
        lr = lr_min + 1/2 * (1 + cos(cos_inner)) * (lr_max - lr_min)
    return lr