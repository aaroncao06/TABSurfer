
import subprocess
import os
import pathlib
import argparse
import pandas as pd

'''
python TABSurfer_recon_aseg_batch.py --input_dir /media/sail/HDD24T/DeepContrast/DeepContrast_JG_Generating_AICBV_from_MNI152_iso1mm_T1w_Pretrained_3D_Model/Dataset/TABSurfer_demo_T1w/T1w_multiple_studies/ --output_dir /media/sail/HDD24T/DeepContrast/DeepContrast_JG_Generating_AICBV_from_MNI152_iso1mm_T1w_Pretrained_3D_Model/Dataset/TABSurfer_demo_T1w/
'''
parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', default='', type=str) 
parser.add_argument('--output_dir', default='processed', type=str) 
parser.add_argument('--mgz', default=False, type=bool) 

parser.add_argument('--model_path', default=os.path.join(os.path.dirname(__file__), 'model_checkpoints', 'TABSurfer_ISBI_final.pt'),
                    type=str)

parser.add_argument('--gpu_available', default=True, type=bool) 
parser.add_argument('--gpu_id', default=0, type=int) 

parser.add_argument('--step_size', default=16, type=int)

args = parser.parse_args()

if __name__ == '__main__':
  input_dir = args.input_dir
  output_dir = args.output_dir
  mgz = args.mgz
  model_path = args.model_path
  gpu_available = args.gpu_available
  gpu_id = args.gpu_id
  step_size = args.step_size

  os.makedirs(f'{output_dir}/conformed', exist_ok=True)
  os.makedirs(f'{output_dir}/segmented', exist_ok=True)

  for files in os.listdir(input_dir):
    if files.endswith('.nii.gz') or files.endswith('.mgz'):
      #print(files)
      img_path = os.path.join(input_dir, files)
      file_name_split = os.path.splitext(files)
      file_name_split = file_name_split[0]
      file_name_split = os.path.splitext(file_name_split)
      file_name = file_name_split[0]
      if file_name == 'ANTs_XYF_Aging_18_to_23_T1w_template':
        continue
      print('********************************************', flush = True)
      print('********************************************', flush = True)
      print(file_name, flush = True)
      print('********************************************', flush = True)
      print('********************************************', flush = True)
      #print(os.path.exists(img_path))
      output_path = None
      conformed_path = None
      if mgz:
        output_path = os.path.join(output_dir, f'segmented/aseg.{file_name}.mgz')
        conformed_path = os.path.join(output_dir, f'conformed/orig.{file_name}.mgz')
      else:
        output_path = os.path.join(output_dir, f'segmented/aseg.{file_name}.nii.gz')
        conformed_path = os.path.join(output_dir, f'conformed/orig.{file_name}.nii.gz')
      if os.path.exists(output_path) and os.path.exists(conformed_path):
        print('scan already processed before', flush=True)
        #continue
      new_args = ["--input_path", img_path, 
              "--output_T1_path", conformed_path,
              '--output_aseg_path', output_path,
              '--model_path', model_path,
              '--gpu_available', str(gpu_available),
              '--gpu_id', str(gpu_id),
              '--step_size', str(step_size)]

      # Run the sub_script.py with arguments using subprocess
      subprocess.run(["python", "TABSurfer_recon_aseg.py"] + new_args)

      #ipdb.set_trace()
  print('done', flush = True)
