sudo apt install libvips-tools

https://iipimage.sourceforge.io/documentation/images

vips im_vips2tiff <source_image> <output_image.tif>:<compression>,tile:<size>,pyramid

vips im_vips2tiff data/Visium_HD_Human_Lung_Cancer_tissue_image.tif data/pyramidal.tif:deflate,tile:256x256,pyramid

