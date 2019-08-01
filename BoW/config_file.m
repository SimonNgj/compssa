%%%%% Configuration file %%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DIRECTORIES - please change if copying the code to a new location
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Choose one skill
skill = 'Knot'; 
%skill = 'Needle';
%skill = 'Suture'; 

%% Directory holding all the experiment image frames
DATA_DIR = '../../data_Mar8/data/';

%% Directory holding the codebook, i.e., the k-mean clusters
CODEBOOK_DIR = '../data/codebook/';

%% Feature directory - holds all features
FEATURE_DIR = '../data/feature/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% dataset parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% number of classes 
num_class = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% bag-of-words parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Transform
transform = 'wavelet';

%% length of subsequences
sub_length = 8;

%% over points between slideing window
inter_point = 3;

segment_size = 180;
step = 60;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% codebook parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% size of codebook
VQ.Codebook_Size = 100;
%% Max number of k-means iterations
VQ.Max_Iterations = 25;
%% Verbsoity of Mark's code
VQ.Verbosity = 0;