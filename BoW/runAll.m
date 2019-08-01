
%% Combine all together

config_file = 'config_file';
try
    eval(config_file);
catch
    disp('config file failed');
end

1_encode;
2_codebook;
3_histo;