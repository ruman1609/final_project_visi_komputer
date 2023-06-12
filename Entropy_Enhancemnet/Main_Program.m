%% In this program, we will remove the unnecessary details from images.

clear; close all; clc

% Reading Images and saving all of them in a single
% folder after resizing.
dir_path = '../Dataset/1_asli';
dir_save_path = '../Dataset/2_0_enhanced';
% 
% Reading all the images from the directory
img_names = dir([dir_path, '/*.jpeg']);
disp(['There are ', num2str(length(img_names)),' Images'])
for ind = 1:length(img_names)
    disp(strcat('Processing... ', img_names(ind).name))
    image = imread(fullfile(dir_path, img_names(ind).name));
    image = imresize(image,[256,256]);
    [J,~,~] = entropy_enhancement(image);
    if isempty(J)
        disp('skipped.................')
        continue
    end
    save_path = fullfile(dir_save_path, img_names(ind).name);
    imwrite(J, save_path)
end
