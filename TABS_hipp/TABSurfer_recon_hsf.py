import torch
import numpy as np
import gc
import os
import nibabel as nib
import random
import argparse
from numpy.typing import NDArray
import torch.nn.functional as F
import sys
from Models.TABS_Model import TABS_hipp
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from TABSurfer_aseg.Models.TABS_Model import TABS_new
from TABSurfer_aseg.conform import rescale, conform, is_conform

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default = 1000, type=int)

parser.add_argument('--input_path', default='', type=str) 
parser.add_argument('--output_T1_path', default='', type=str) 
parser.add_argument('--output_aseg_path', default='', type=str)
parser.add_argument('--output_hsf_path', default = '', type=str) 
parser.add_argument('--aseg_model_path', default=
                    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TABSurfer_aseg', 'model_cheeckpoints', 'TABSurfer_ISBI_final.pt')), 
                    type=str)
parser.add_argument('--hsf_model_path', default = 
                    os.path.join(os.path.dirname(__file__), 'model_checkpoints', 'TABSurfer_hsf.pt'), 
                    type=str)

parser.add_argument('--gpu_id', default=0, type=int) 

parser.add_argument('--aseg_step_size', default=16, type=int)
args = parser.parse_args()

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

def run_and_reconstruct(model, num_classes, image: NDArray, patch_size=96, step=16, device = torch.device('cuda')):
    
    (H, W, D) = image.shape
    reconstructed = torch.zeros((num_classes, H, W, D))
    
    model.eval()
    for patch, coordinate, count in iter(sequential_patch_iter(image, patch_size, step)):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del count
        (x, y, z) = coordinate
        patch = torch.from_numpy(patch).float().squeeze().unsqueeze(0).unsqueeze(0)
        patch = patch.to(device)
        predicted_patch = None
        with torch.no_grad():
            predicted_patch = model(patch).cpu() # 1 1 96 96 96
            del patch
        #add the voted probability
        reconstructed[:, x : x + patch_size, y : y + patch_size, z : z + patch_size] = reconstructed[:, x : x + patch_size, y : y + patch_size, z : z + patch_size] + predicted_patch.squeeze()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reconstructed = reconstructed.squeeze()
    reconstructed = F.softmax(reconstructed, dim=0)
    return reconstructed.numpy()

def before_and_after(array):
    before = np.argmax(array)
    after = len(array)-np.sum(array)-before
    return before, after

def crop_images_full_flush_single(conformed_img , step_size): #158, 170, 202 max raw

    valid_slices_x = ~np.all(conformed_img == 0, axis=(1, 2))
    valid_slices_y = ~np.all(conformed_img == 0, axis=(0, 2))
    valid_slices_z = ~np.all(conformed_img == 0, axis=(0, 1))

    pad_before_x, pad_after_x = before_and_after(valid_slices_x)
    pad_before_y, pad_after_y = before_and_after(valid_slices_y)
    pad_before_z, pad_after_z = before_and_after(valid_slices_z)

    # Crop the array to remove zero slices along each dimension
    cropped_conformed_img = conformed_img[valid_slices_x][:, valid_slices_y][:, :, valid_slices_z]
    
    original_shape = cropped_conformed_img.shape
    #print(original_shape)
    (depth_pad, height_pad, width_pad) = tuple((step_size-((dim-96) % step_size)) for dim in original_shape)
    
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
    
    final_conformed_img = np.pad(cropped_conformed_img, ((depth_pad_before, depth_pad_after),
                                                           (height_pad_before, height_pad_after),
                                                           (width_pad_before, width_pad_after)),
                                   mode='constant', constant_values=0)
    
    padding = ((pad_before_x, pad_after_x),(pad_before_y, pad_after_y),(pad_before_z, pad_after_z))
    #return conformed_img, ((0,0),(0,0),(0,0))
    return final_conformed_img, padding

def map_fs_aseg_labels(volume, orig_mapping_dict = {0: 0, 4: 1, 5: 2, 7: 3, 8: 4, 10: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11, 17: 12, 18: 13, 24: 14, 26: 15, 28: 16, 31: 17, 43: 18, 44: 19, 46: 20, 47: 21, 49: 22, 50: 23, 51: 24, 52: 25, 53: 26, 54: 27, 58: 28, 60: 29, 63: 30, 77: 31}):
    #valid_classes = [0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63, 77]
    #mapping = {value: index for index, value in enumerate(valid_classes)}
    reversed_mapping_dict = {value: key for key, value in orig_mapping_dict.items()}
    mapped_volume = np.vectorize(reversed_mapping_dict.get)(volume)
    return mapped_volume
def map_fs_hsf_labels(volume, FS60_hipp_classes = [0, 203, 204, 205, 206, 208, 209, 210, 211, 212, 214, 215, 226]):
    mapping = {index : value for index, value in enumerate(FS60_hipp_classes)}
    mapped_volume = np.vectorize(mapping.get)(volume)
    return mapped_volume

def run_aseg(aseg_model, input_img_path, aseg_save_path = '', conformed_input_save_path = '', step_size = 16, device = torch.device('cuda')):
    print(f'step size {step_size}')
    fs_affine_matrix = np.array([
        [-1.0, 0.0, 0.0, 127.0],
        [0.0, 0.0, 1.0, -145.0],
        [0.0, -1.0, 0.0, 147.0],
        [0.0, 0.0, 0.0, 1.0]])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print()
    print(input_img_path, flush=True)
    if not os.path.exists(input_img_path):
        return
    
    input_img = nib.load(input_img_path)
    conformed_input_img = None
    if not is_conform(input_img, verbose = False):
        print('conform', flush=True)
        input_img = conform(input_img)
    if os.path.exists(conformed_input_save_path):
        nib.save(input_img, conformed_input_save_path)
    input_img = input_img.get_fdata()
    print(np.min(input_img))
    print(np.max(input_img))
    print('conformed, rescale intensity', flush=True)
    scaled_raw_conformed_img = rescale(input_img, 0, 255)
    scaled_raw_conformed_img[input_img==0]==0
    conformed_input_img = np.uint8(np.rint(scaled_raw_conformed_img)) / 255 # convert to uchar like in conform and then scale between 0 and 1
    del scaled_raw_conformed_img

    #ipdb.set_trace()
    cropped_conformed_img, padding = crop_images_full_flush_single(conformed_input_img, step_size)
    #ipdb.set_trace()
    print(cropped_conformed_img.shape, flush=True)
    print(padding, flush=True)
    print('run inference', flush=True)
    full_predicted_scan = run_and_reconstruct(aseg_model, num_classes = 32, image = cropped_conformed_img, patch_size=96, step=step_size, device = device)
    print('save segmentation', flush=True)
    full_predicted_scan = np.argmax(full_predicted_scan, axis = 0)
    
    padded_full_predicted_scan = np.pad(full_predicted_scan, padding, mode='constant', constant_values=0)
    print(padded_full_predicted_scan.shape, flush=True)
    padded_full_predicted_scan = map_fs_aseg_labels(padded_full_predicted_scan)
    aseg = nib.Nifti1Image(padded_full_predicted_scan, affine=fs_affine_matrix)
    if os.path.exists(aseg_save_path):
        nib.save(aseg, aseg_save_path)
    print('done', flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return aseg, input_img
def get_cropped_input(aseg, input_img):
    valid_aseg_classes = [17, 53]
    input_aseg_masked = np.isin(aseg, valid_aseg_classes)

    #get bounding box 96 x 96 x 96
    valid_slices_x = ~np.all(input_aseg_masked == 0, axis=(1, 2)).squeeze()
    valid_slices_y = ~np.all(input_aseg_masked == 0, axis=(0, 2)).squeeze()
    valid_slices_z = ~np.all(input_aseg_masked == 0, axis=(0, 1)).squeeze()

    first_x = np.where(valid_slices_x == 1)[0][0]
    last_x = np.where(valid_slices_x == 1)[0][-1]

    first_y = np.where(valid_slices_y == 1)[0][0]
    last_y = np.where(valid_slices_y == 1)[0][-1]

    first_z = np.where(valid_slices_z == 1)[0][0]
    last_z = np.where(valid_slices_z == 1)[0][-1]


    x_pad = 96 - (last_x - first_x + 1)
    x_pad_before = x_pad//2
    x_pad_after = x_pad - x_pad_before

    valid_slices_x[first_x - x_pad_before : last_x + x_pad_after + 1] = 1

    y_pad = 96 - (last_y - first_y + 1)
    y_pad_before = y_pad//2
    y_pad_after = y_pad - y_pad_before
    
    valid_slices_y[first_y - y_pad_before : last_y + y_pad_after + 1] = 1

    z_pad = 96 - (last_z - first_z + 1)
    z_pad_before = z_pad//2
    z_pad_after = z_pad - z_pad_before

    valid_slices_z[first_z - z_pad_before : last_z + z_pad_after + 1] = 1

    cropped_img = input_img[valid_slices_x][:, valid_slices_y][:, :, valid_slices_z]
    
    padding_x = (first_x - x_pad_before, 255 - last_x - x_pad_after)
    padding_y = (first_y - y_pad_before, 255 - last_y - y_pad_after)
    padding_z = (first_z - z_pad_before, 255 - last_z - z_pad_after)
    padding = (padding_x, padding_y, padding_z)
    return torch.from_numpy(cropped_img.squeeze()).unsqueeze(0).unsqueeze(0), padding
def run_hsf(hsf_model, aseg, input_img, hsf_save_path = '', device = torch.device('cuda')):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fs_affine_matrix = np.array([
        [-1.0, 0.0, 0.0, 127.0],
        [0.0, 0.0, 1.0, -145.0],
        [0.0, -1.0, 0.0, 147.0],
        [0.0, 0.0, 0.0, 1.0]])
    cropped_input, padding = get_cropped_input(aseg, input_img)
    cropped_input = cropped_input.float().cuda()
    with torch.no_grad():
        predicted_patch = hsf_model(cropped_input).cpu()
    predicted_patch = np.argmax(np.array(predicted_patch.squeeze()), axis = 0).squeeze()
    full_predicted_scan = np.pad(predicted_patch, padding, mode='constant', constant_values=0)
    full_predicted_scan = map_fs_hsf_labels(full_predicted_scan)
    seg = nib.Nifti1Image(full_predicted_scan, affine=fs_affine_matrix)
    if os.path.exists(hsf_save_path):
        nib.save(seg, hsf_save_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return seg


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if (args.gpu_id != None) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    aseg_model = TABS_new(img_dim = 96,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 32,
        embedding_dim = 1024,
        num_heads = 16,
        num_layers = 8,
        dropout_rate = 0.1,
        attn_dropout_rate = 0.1)
    aseg_model.load_state_dict(torch.load(args.aseg_model_path, map_location=device))
    print('run aseg')
    aseg, input_img = run_aseg(aseg_model, args.input_path, aseg_save_path = args.output_aseg_path, conformed_input_save_path = args.output_T1_path, step_size = 16, device=device)
    del aseg_model
    hsf_model = TABS_hipp()
    hsf_model.load_state_dict(torch.load(args.hsf_model_path, map_location=device))
    print('run hsf')
    hsf = run_hsf(hsf_model, aseg.get_fdata(), input_img, hsf_save_path = args.output_hsf_path, device=device)
    del hsf_model
    print('done')
    #...
    