import numpy as np
from numpy.typing import NDArray
from Models.TABS_Model import TABS_new
import torch
import gc
import nibabel as nib
import argparse
import random
from tqdm import tqdm
import ipdb
from conform import conform, is_conform
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default = 1000, type=int)

parser.add_argument('--input_path', default='', type=str) 
parser.add_argument('--output_T1_path', default='', type=str) 
parser.add_argument('--output_aseg_path', default='', type=str) 

parser.add_argument('--model_path', default=
                    os.path.join(os.path.dirname(__file__), 'model_checkpoints', 'TABSurfer_ISBI_final.pt'), 
                    type=str)

parser.add_argument('--gpu_available', default=True, type=bool) 
parser.add_argument('--gpu_id', default=0, type=int) 

parser.add_argument('--step_size', default=32, type=int)


'''input path 
output path 
model path 
step size
output format mgz vs nii.gz

option to have output match shape of input for overlay'''


args = parser.parse_args()

def before_and_after(array):
    before = np.argmax(array)
    after = len(array)-np.sum(array)-before
    return before, after
def pad_or_crop_array(array, padding):
    slices = []
    for i, (pad_before, pad_after) in enumerate(padding):
        shape = array.shape[i]
        start = max(0, -pad_before)
        end = shape - max(0, -pad_after)
        slices+=[start, end]
    
    cropped_array = array[slices[0]:slices[1],slices[2]:slices[3],slices[4]:slices[5]]
    
    final_padding = []
    for pad_before, pad_after in padding:
        pad_before = max(0, pad_before)
        pad_after = max(0, pad_after)
        final_padding.append((pad_before, pad_after))
    
    padded_array = np.pad(cropped_array, final_padding, mode='constant', constant_values=0)
    return padded_array
def crop_image_full_flush(conformed_img, step_size): #158, 170, 202 max raw

    valid_slices_x = ~np.all(conformed_img == 0, axis=(1, 2))
    valid_slices_y = ~np.all(conformed_img == 0, axis=(0, 2))
    valid_slices_z = ~np.all(conformed_img == 0, axis=(0, 1))

    pad_before_x, pad_after_x = before_and_after(valid_slices_x)
    pad_before_y, pad_after_y = before_and_after(valid_slices_y)
    pad_before_z, pad_after_z = before_and_after(valid_slices_z)

    # Crop the array to remove zero slices along each dimension
    cropped_conformed_img = conformed_img[valid_slices_x][:, valid_slices_y][:, :, valid_slices_z]
    
    original_shape = cropped_conformed_img.shape

    (depth_pad, height_pad, width_pad) = tuple((step_size - ((dim-96) % step_size)) for dim in original_shape)

    depth_pad_before = depth_pad // 2
    depth_pad_after = depth_pad - depth_pad_before
    
    pad_before_x -= depth_pad_before
    pad_after_x -= depth_pad_after

    height_pad_before = height_pad // 2
    height_pad_after = height_pad - height_pad_before

    pad_before_y -= height_pad_before
    pad_after_y -= height_pad_after

    width_pad_before = width_pad // 2
    width_pad_after = width_pad - width_pad_before

    pad_before_z -= width_pad_before
    pad_after_z -= width_pad_after

    final_conformed_img = pad_or_crop_array(cropped_conformed_img, ((depth_pad_before, depth_pad_after),
                                                           (height_pad_before, height_pad_after),
                                                           (width_pad_before, width_pad_after)))
    # final_conformed_img = np.pad(cropped_conformed_img, ((depth_pad_before, depth_pad_after),
    #                                                        (height_pad_before, height_pad_after),
    #                                                        (width_pad_before, width_pad_after)),
    #                                mode='constant', constant_values=0)

    padding = ((pad_before_x, pad_after_x),(pad_before_y, pad_after_y),(pad_before_z, pad_after_z))
    return final_conformed_img, padding

def sequential_patch_iter(image: NDArray, patch_size=96, step=16):

    (H, W, D) = image.shape
    count=0
    #image_zeropadding = patch/2
    for z in range(0, D - patch_size+1, step):
        for y in range(0, W- patch_size+1, step):
            for x in range(0, H - patch_size+1, step):

                patch = np.float32(image[x : x + patch_size, y : y + patch_size, z : z + patch_size])

                coordinate = (x, y, z)
                count=count+1
                
                yield patch.squeeze(), coordinate, count #count starts from 1

def run_and_reconstruct(model, num_classes, image: NDArray, patch_size=96, step=16, cuda = False):
    (H, W, D) = image.shape
    reconstructed = np.zeros((num_classes, H, W, D))
    if cuda:
        model.cuda()
    model.eval()
    for patch, coordinate, count in tqdm(iter(sequential_patch_iter(image, patch_size, step))):
        gc.collect()
        torch.cuda.empty_cache()
        del count
        (x, y, z) = coordinate
        patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
        if cuda:
            patch = patch.cuda()
        predicted_patch = None
        with torch.no_grad():
            predicted_patch = model(patch).cpu() # 1 1 96 96 96
            del patch
        #add the voted probability
        reconstructed[:, x : x + patch_size, y : y + patch_size, z : z + patch_size] = reconstructed[:, x : x + patch_size, y : y + patch_size, z : z + patch_size] + predicted_patch.numpy().squeeze()
    return np.argmax(reconstructed, axis = 0) #find greatest probability class

def map_free_surfer_labels(volume, orig_mapping_dict = {0: 0, 4: 1, 5: 2, 7: 3, 8: 4, 10: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11, 17: 12, 18: 13, 24: 14, 26: 15, 28: 16, 31: 17, 43: 18, 44: 19, 46: 20, 47: 21, 49: 22, 50: 23, 51: 24, 52: 25, 53: 26, 54: 27, 58: 28, 60: 29, 63: 30, 77: 31}):
    reversed_mapping_dict = {value: key for key, value in orig_mapping_dict.items()}
    mapped_volume = np.vectorize(reversed_mapping_dict.get)(volume)
    return mapped_volume

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if (args.gpu_id != None) and (args.gpu_available):
        torch.cuda.set_device(args.gpu_id)
    print('loading image', flush=True)
    input_img = nib.load(args.input_path)
    #ipdb.set_trace()
    print('conform', flush=True)
    t1 = None
    if is_conform(input_img):
        t1 = input_img
    else:
        t1 = conform(input_img)
    print(t1.shape, flush=True)
    print('save conform', flush=True)
    t1_filename = args.output_T1_path
    if t1_filename == '':
        print('dont save conformed input', flush=True)
    else:
        nib.save(t1, t1_filename)
    print('fetch model', flush=True)
    model = TABS_new(img_dim = 96,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 32,
        embedding_dim = 1024,
        num_heads = 16,
        num_layers = 8,
        dropout_rate = 0.1,
        attn_dropout_rate = 0.1)
    #ipdb.set_trace()
    checkpoint_path = args.model_path
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    print('load and crop conformed', flush=True)
    t1_img = t1.get_fdata() / 255
    #ipdb.set_trace()
    cropped_t1_img, padding = crop_image_full_flush(t1_img, args.step_size)
    #ipdb.set_trace()
    print(cropped_t1_img.shape, flush=True)
    print(padding, flush=True)
    print('run inference', flush=True)
    full_predicted_scan = run_and_reconstruct(model, num_classes = 32, image = cropped_t1_img, patch_size=96, step=args.step_size, cuda = args.gpu_available)
    print('save segmentation', flush=True)
    padded_full_predicted_scan = pad_or_crop_array(full_predicted_scan, padding)
    # padded_full_predicted_scan = np.pad(full_predicted_scan, padding, mode='constant', constant_values=0)
    print(padded_full_predicted_scan.shape, flush=True)

    padded_full_predicted_scan = map_free_surfer_labels(padded_full_predicted_scan)

    #ipdb.set_trace()
    fs_affine_matrix = np.array([
        [-1.0, 0.0, 0.0, 127.0],
        [0.0, 0.0, 1.0, -145.0],
        [0.0, -1.0, 0.0, 147.0],
        [0.0, 0.0, 0.0, 1.0]])
    seg = nib.Nifti1Image(np.uint16(padded_full_predicted_scan), affine=fs_affine_matrix)
    seg_filename = args.output_aseg_path
    nib.save(seg, seg_filename)
    print('done', flush=True)