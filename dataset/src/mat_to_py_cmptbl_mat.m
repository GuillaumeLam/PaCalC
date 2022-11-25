% script for data.mat => py_compatible_data.mat
% todo: get relevant info from data.mat

gait_ireg_surface_dataset = smplfd_dataset();

save('../py_compatible_data.mat', 'gait_ireg_surface_dataset');
