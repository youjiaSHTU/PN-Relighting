import os
import sys
sys.path.append('.')

### ------------ config ----------------
import argparse
parser = argparse.ArgumentParser(description='path')
parser.add_argument('--folder', type=str, default = 'test_data', help = 'Path of workspace folder')

### ---------- output path -------------
args = parser.parse_args()
workspace_folder = args.folder

input_img_folder       = os.path.join(workspace_folder, 'img')
input_mask_folder      = os.path.join(workspace_folder, 'mask')
input_relit_env_folder = os.path.join(workspace_folder, 'env')
output_normal_folder   = os.path.join(workspace_folder, 'normal')
output_albedo_folder   = os.path.join(workspace_folder, 'albedo')
output_relit_folder    = os.path.join(workspace_folder, 'relight')
### ------------------------------------

### ------------ model -----------------

normal_model = 'checkpoints/normal'
albedo_model = 'checkpoints/albedo'
relit_model  = 'checkpoints/relight'

### ------------------------------------

### --------- test code path -----------

normal_test_path = './eval/normal_test.py'
albedo_test_path = './eval/albedo_test.py'
relit_test_path  = './eval/relit_test.py'

### ------------------------------------

### run 

print('gen normal...')
os.system('python ' + normal_test_path + ' --pic ' + input_img_folder + ' --mask ' + input_mask_folder + \
	' --out ' + output_normal_folder + ' --cp ' + normal_model)

print('gen albedo...')
os.system('python ' + albedo_test_path + ' --pic ' + input_img_folder + ' --mask ' + input_mask_folder + \
	' --normal ' + output_normal_folder + ' --out ' + output_albedo_folder + ' --cp ' + albedo_model)

print('gen relighting...')
os.system('python ' + relit_test_path + ' --pic ' + input_img_folder + ' --mask ' + input_mask_folder + ' --normal ' + output_normal_folder + \
	' --albedo ' + output_albedo_folder + ' --env ' + input_relit_env_folder + ' --out ' + output_relit_folder + ' --cp ' + relit_model)

print('done!')

