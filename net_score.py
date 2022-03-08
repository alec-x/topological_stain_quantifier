import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.measure import block_reduce
from scipy import ndimage
from argparse import ArgumentParser as arg_parser
from matplotlib import pyplot as plt

def block_calc_2D(in_shp, trgt_shp):
    return (int(in_shp[0]/trgt_shp[0]), int(in_shp[1]/trgt_shp[1]))

def calc_granularity(arr, out_shp):
    sobel_img = cv2.Sobel(arr, cv2.CV_64F, dx=1, dy=1, ksize=7)
    sobel_img = sobel_img / np.max(sobel_img)

    gran_steps = []
    gran_steps.append(block_reduce(sobel_img, (16,16), np.max)) 
    gran_steps.append(ndimage.gaussian_filter(gran_steps[-1], 1))
    # normalize med filter results
    gran_steps.append(ndimage.median_filter(gran_steps[-1], 10)/100)
    blck_shp = block_calc_2D(gran_steps[-1].shape, out_shp)
    gran_steps.append(block_reduce(gran_steps[-1], blck_shp, np.mean))

    return gran_steps[-1]

def calc_brightness(arr, out_shp):
    bright_steps = []
    bright_steps.append(block_reduce(arr, (16,16), np.mean)) 
    bright_steps.append(ndimage.gaussian_filter(bright_steps[-1], 1))
    bright_steps.append(ndimage.median_filter(bright_steps[-1], 10)/100)
    blck_shp = block_calc_2D(bright_steps[-1].shape, out_shp)
    bright_steps.append(block_reduce(bright_steps[-1], blck_shp, np.mean))
    
    return bright_steps[-1]

def calc_density(arr, out_shp):
    dense_steps = []
    dense_steps.append(block_reduce(arr, (16,16), np.mean)) 
    dense_steps.append(ndimage.gaussian_filter(dense_steps[-1], 1))
    dense_steps.append(ndimage.median_filter(dense_steps[-1], 10)/100)
    blck_shp = block_calc_2D(dense_steps[-1].shape, out_shp)
    dense_steps.append(block_reduce(dense_steps[-1], blck_shp, np.mean))
    
    return dense_steps[-1]

def main(raw_args=None):
    parser = arg_parser(description="Change august to december format")
    parser.add_argument("egfp_path", action="store", type=str, \
                        help="2D EGFP channel image")
    parser.add_argument("cy5_path", action="store", type=str, \
                        help="2D Cy5 channel image")
    parser.add_argument("dim_0", action="store", type=int, \
                        help="Pixels in output dimension 0")
    parser.add_argument("dim_1", action="store", type=int, \
                        help="Pixels in output dimension 1")
    parser.add_argument("-weight_b", action="store", type=int, \
                        help="Weight placed on brightness in calculating NET \
                              score (default 1.0)")
    parser.add_argument("-weight_g", action="store", type=int, \
                        help="Weight placed on granularity in calculating NET \
                              score (default 1.0)")                              
    parser.add_argument("-o", action="store", type=str, \
                        help="out path of result (default=current dir)")
    parser.add_argument("-s", action="store_true", \
                        help="Show output comparison (default=false)")
    parser.set_defaults(o=".\output.png", s=False, weight_b=1.0, weight_g=1.0)
    args = parser.parse_args()

    out_dims = (args.dim_0, args.dim_1)
    img_temp = Image.open(args.egfp_path)
    egfp_arr = np.array(img_temp)
    img_temp = Image.open(args.cy5_path)
    cy5_arr = np.array(img_temp)

    granularity = calc_granularity(egfp_arr, out_dims)
    brightness = calc_brightness(egfp_arr, out_dims)

    

    density = calc_density(cy5_arr, out_dims)

    brightness = args.weight_b*brightness/np.max(brightness)
    # take inverse of granularity 
    granularity = args.weight_g*(1-granularity/np.max(granularity))
    density = density/np.max(density)
    net_score = np.multiply(brightness, granularity, density)
    
    if args.s:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        fig.suptitle('Vertically stacked subplots')
        
        ax1.imshow(granularity)
        ax1.set_title("Granularity (EGFP)")
        
        ax2.imshow(brightness)
        ax2.set_title("Brightness (EGFP)")
        
        ax3.imshow(density)
        ax3.set_title("Density (EGFP)")
        
        ax4.imshow(net_score)
        ax4.set_title("NET Score (higher density brighter)")

        for ax in fig.get_axes():
            ax.label_outer()
        plt.show()

    img_out = Image.fromarray(net_score)
    img_out.convert('RGB').save(args.o)

if __name__=="__main__":
    main()