%% Created: Xuan Anh Nguyen  03/10/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
config_file = 'config_file_1';
try
    eval(config_file);
catch
    disp('config file failed');
end

%% load the feature_data
load([CODEBOOK_DIR,strcat(skill,'_data_encode_',num2str(segment_size),'_', num2str(step),'.mat')]);% use the standarlized features

%% descriptors for the k-means, only use several sequence for each class
% generate random index
for u_out =1:5
    all_descriptors = [];
    for i = 1:size(data_tran,2) % i from 1 to 3
        temp_index = randperm(size(data_tran{i},2));  
        tj = 1;ti = 1;
        while tj <= fix(size(temp_index,2)/10) % only use a subset (1/10 here) of training data to construct the codebook:
            if data_tran{i}{3,ti} == u_out
                ti = ti + 1; 
            else
                all_descriptors = [all_descriptors,data_tran{i}{1,temp_index(ti)}];
                tj = tj + 1;
                ti = ti + 1;
            end      
        end
    end
    %name = strcat(skill,'_data_tran.mat');
    %clear name; % save memory

    %% codebook size
    codebook_size = VQ.Codebook_Size;

    %% form options structure for clustering
    cluster_options.maxiters = VQ.Max_Iterations;
    cluster_options.verbose  = VQ.Verbosity;

    %% OK, now call kmeans clustering: 100 clusters, each cluster has 66 dimensions
    [centers,sse] = vgg_kmeans(double(all_descriptors), codebook_size, cluster_options);

    %% form name to save codebook
    du_mkdir(CODEBOOK_DIR);
    fname = [CODEBOOK_DIR ,skill,num2str(segment_size),'_', num2str(step),'_kmean_', num2str(codebook_size),'_',num2str(u_out),'.mat']; 

    %% save centers to file...
    save(fname,'centers');
end