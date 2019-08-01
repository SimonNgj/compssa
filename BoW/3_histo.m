%% Created: Xuan Anh Nguyen  05/09/2018 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
config_file = 'config_file_1';
try
    eval(config_file);
catch
    disp('config file failed');
end

%% Function vector-quantizes the features using the codebook
%% A histogram over codebook entries is also computed and stored.            
        
%% load the string data
load([CODEBOOK_DIR,strcat(skill,'_data_encode_',num2str(segment_size),'_', num2str(step),'.mat')]);

for u_out = 1:5
    load([CODEBOOK_DIR,strcat(skill,num2str(segment_size),'_', num2str(step),'_kmean_', num2str(VQ.Codebook_Size),'_',num2str(u_out),'.mat')]);

    %% loop over all features
    for i=1:size(data_tran,2) % i from 1 to 3
    %     disp(i);
        for j = 1:size(data_tran{i},2) 
         
            %% Find number of points per time series
            nPoints = size(data_tran{i}{1,j},2); 
    
            %% Set distance matrix to all be large values
            distance = Inf * ones(nPoints,VQ.Codebook_Size); 
    
            %% Loop over all centers and all points and get L2 norm btw. the two.
            [row,col] = size(data_tran{i}{1,j});
            centers = reshape(centers,[row,1,VQ.Codebook_Size]);
            centersMat = repmat(centers,[1,col,1]);
            dataMat = repmat(data_tran{i}{1,j},[1,1,VQ.Codebook_Size]);
            dataMat = (dataMat - centersMat).^2;
            dataSum = sum(dataMat,1);
            distance = reshape(dataSum,[],size(dataSum,3));% we do not need to .^0.5
        
            %% Now find the closest center for each point
            [tmp,descriptor_vq] = min(distance,[],2); 

            %% Now compute histogram over codebook entries for song
            histogram = zeros(1,VQ.Codebook_Size);
            for p = 1:nPoints 
                histogram(descriptor_vq(p)) = histogram(descriptor_vq(p)) + 1;      
            end
        
            feature_hist{i}{1,j} = histogram'; 
            feature_hist{i}{2,j} = data_tran{i}{2,j};
            feature_hist{i}{3,j} = data_tran{i}{3,j};
        end
    end
    
    %% save bag-of-words representation
    savepath = [FEATURE_DIR,skill,num2str(segment_size),'_', num2str(step),'_FeatureHist_',num2str(sub_length),'_',num2str(VQ.Codebook_Size),'_',num2str(u_out),'.mat'];
    save(savepath,'feature_hist');
end
