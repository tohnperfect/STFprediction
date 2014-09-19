%list files
working_folder='';%%%path to the folder that contains images e.g. '/home/localadmin/dataset/'
list=dir(strcat(working_folder,'images/')); 
mkdir('features');

% Parameters:
model_dir = './models/CNN_S';

param_file = sprintf('%s/param.prototxt', model_dir);
model_file = sprintf('%s/model', model_dir);

average_image = './models/mean.mat';
    
encoder = featpipem.directencode.ConvNetEncoder(param_file, model_file, ...
                                                average_image, ...
                                                'output_blob_name', 'fc7');
    
encoder.augmentation = 'aspect_corners';
encoder.augmentation_collate = 'none';

for i=3:numel(list)-1
    
    fname=list(i,1).name;
    im=imread(strcat(strcat(working_folder,'images/'),fname));
    
    

                                            
    im = featpipem.utility.standardizeImage(im); % ensure of type single, w. three channels

   
    code_augmented = encoder.encode(im);

    cov=reshape(code_augmented,1,40960);
    
    sname=fname(1:numel(fname)-4);
    save(strcat(strcat(working_folder,'features/'),sname),'cnn');
    
    fprintf(sname)
    fprintf('\n')
    
end
