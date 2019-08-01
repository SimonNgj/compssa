function feature_vector = Extract_basic_features (data)
n = size(data, 1);
var_num = size(data, 2);
window = size(data, 3);
feature_vector = [];

for i = 1: n
    features = [];
    for j = 1:var_num
        seg = data(i,j,:);
        % 4 features
        features = [features,mean(seg),std(seg),max(seg),min(seg)];
        % 2 features
        features = [features,rssq(seg)/window,rssq(seg-mean(seg))/n];
        % 1 features
        features = [features,mean(abs(seg - mean(seg)))];
        % 10 features
        s1 = seg(:); 
        features = [features,hist(s1, 10)/window];
    end
    
    % 1 features
    features = [features,mean2(data(i,:,:))];
    
    %% features vector
    feature_vector = [feature_vector; features];
end

end