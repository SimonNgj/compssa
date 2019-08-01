%% Created: Xuan Anh Nguyen  03/10/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
config_file = 'config_file_1';
try
    eval(config_file);
catch
    disp('config file failed');
end

%% Load data
load([DATA_DIR,strcat(skill,'_dataX_basic_',num2str(segment_size),'_', num2str(step),'.mat')]);

for i = 1:size(dataX,2) % i = 1:3
    for j = 1:size(dataX{i},2) 
        %% sliding a subwindows along a time series and extract a feature
        %% vector for each sub sequences by transformation (e.g., 'wavelet')
        data_tran{i}{1,j} = series_transform(dataX{i}{1,j}',sub_length,inter_point);
        data_tran{i}{2,j} = dataX{i}{2,j};
        data_tran{i}{3,j} = dataX{i}{3,j};
    end
end
    
du_mkdir(CODEBOOK_DIR);
save([CODEBOOK_DIR,strcat(skill,'_data_encode_',num2str(segment_size),'_', num2str(step),'.mat')],'data_tran');
