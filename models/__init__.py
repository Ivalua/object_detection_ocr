import models.CNN_C32_C64_M2_C128_D
import models.CNN_C32_C64_M2_C64_C64_M2_C128_D
import models.CNN_C32_C64_M2_C64_C64_M2_C128_D_2
import models.CNN_C64_C128_M2_C256_D
import models.CNN_C128_C256_M2_C512_D
import models.CNN_C64_C128_M2_C128_C128_M2_C256_D_2
import models.CNN_C64_C128_M2_C128_C128_M2_C256_D_3
import models.CNN_C128_C256_M2_C256_C256_M2_C512_D_2
import models.CNN_C64_C128_M2_C128_C128_M2_C256_D_2_S7
import models.CNN_C32_C64_C128_D
import models.CNN_C32_C64_C128_C
import models.CNN_C32_C64_C128_C2
import models.CNN_C32_C64_C64_Cd64_C128_D
import models.CNN_C32_Cd64_C64_Cd64_C128_D
import models.vgg
import models.VGG16_D256
import models.VGG16_D4096_D4096
import models.VGG16_block4_D4096_D4096
import models.VGG16_AVG
import models.VGG16_AVG_r
import models.VGG16_C4096_C4096_AVG


def get(**kwargs):
    if kwargs.get("name") not in globals():
        raise KeyError('Unknown network: {}'.format(kwargs))

    return globals()[kwargs.get("name")].Network(kwargs.get("stride_scale"))
