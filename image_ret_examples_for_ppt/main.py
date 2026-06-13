from PIL import Image
# image with pleural effusion and pnemonia and no edema and no lung opacity
img_path = r"/home/tomererez/normal_near_normal/cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/p10/p10000935/s50578979/d0b71acc-b5a62046-bbb5f6b8-7b173b85-65cdf738.jpg"
# save the image
save_path = r"pleural_effusion_and_pnemonia_no_edema_no_lung_opacity.jpg"

# Image.open(img_path).save(save_path)


eff_and_no_edema = r"/home/tomererez/normal_near_normal/cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/p10/10000935/s50578979/d0b71acc-b5a62046-bbb5f6b8-7b173b85-65cdf738.jpg"


#Atelectasis and no Pneumonia 
base = r"/home/tomererez/normal_near_normal/cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/"
# p = base+"p10/p10001884/s56349965"

import os
# print(os.listdir(p))

# a = p+'/1d5dafe8-b4e14a97-e72964db-47f5e168-f3d50666.jpg'
# b = p+'/79863f89-595bf19b-3c7b514a-42969d9b-eff42368.jpg'
# # save

# Image.open(a).save("Atelectasis and no Pneumonia - view A.jpg")
# Image.open(b).save("Atelectasis and no Pneumonia - view B.jpg")



#Atelectasis and Pneumonia 
p = base+"p10/p10003019/s50543252"
print(os.listdir(p))
a = p+'/3f4a324f-7967a6b4-91edf0c8-94fbefd4-32402065.jpg'
b = p+'/e0c97f3f-b7283b86-d58a0da8-8d623549-c695a335.jpg'
# save

Image.open(a).save("Atelectasis and Pneumonia - view A.jpg")
Image.open(b).save("Atelectasis and Pneumonia - view B.jpg")


