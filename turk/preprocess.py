import os
import re

# (100_A_real_A.png, 100_A_real_B.png, 100_A_fake_B.png)
# (0_source.png, 0_real.png, 0_fake.png)
if __name__ == '__main__':
    dir = 'BicycleGAN_googlemap'
    os.chdir(dir)
    for file in os.listdir('.'):
        os.rename(file, dir + '_'+ file)
        # os.rename(file, file.replace("Pix2pix_googlemap", ""))
    # os.chdir('Pix2pix_googlemap')
    # for file in os.listdir('.'):
    #     if re.match(r'.*source', file):
    #         os.rename(file, file.replace("source", "real_A"))
    #     if re.match(r'.*real', file):
    #         os.rename(file, file.replace("real", "real_B"))
    #     if re.match(r'.*fake', file):
    #         os.rename(file, file.replace("fake", "fake_B"))
    # os.chdir('CycleGAN_googlemap')
    # for file in os.listdir('.'):
    #     if re.match(r'.*A_real_A', file):
    #         os.rename(file, file.replace("A_real_A", "real_A"))
    #     if re.match(r'.*A_real_B', file):
    #         os.rename(file, file.replace("A_real_B", "real_B"))
    #     if re.match(r'.*A_fake_B', file):
    #         os.rename(file, file.replace("A_fake_B", "fake_B"))
    # os.chdir('BicycleGAN_googlemap')
    # for file in os.listdir('.'):
    # if re.match(r'.*ground truth', file):
    #     os.rename(file, file.replace("ground truth", "real_B"))
    # if re.match(r'.*A_real_B', file):
    #     os.rename(file, file.replace("A_real_B", "real_B"))
    # if re.match(r'.*sample', file):
    #     os.remove(file)
    # os.rename(file, file.replace("sample05", "fake_B"))
    # if re.match(r'.*A_real_A', file):
    # os.rename(file, file.replace("_random", ""))
