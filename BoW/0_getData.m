%% Created by Xuan Anh Nguyen    03/10/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
config_file = 'config_file_1';
try
    eval(config_file);
catch
    disp('config file failed');
end
dataX = [];
dataY = [];
num_trial = 0;
t1 = 0;
t2 = 0;
t3 = 0;

for userid = 'B':'I'
    for i = 1:5
        if ((userid == 'H') || (userid == 'I'))&&(i == 5) && strcmp(skill, 'Knot')
            continue;
        end
        file_name = strcat('../',skill,'/',skill,'_',userid,'00',num2str(i),'.csv');
        a1 = csvread(file_name);
%		a = a1;
        a = a1(:,39:76);
%         a = zscore(a);
        k = 1;
        while k + segment_size <= size(a,1)
            b = [];
            for j = 1:size(a,2)
                dataX_add = a(k : k + segment_size - 1,j);
                b = [b,dataX_add];
            end
            
            if (userid == 'B') || (userid =='G') || (userid =='H') || (userid =='I')
                t1 = t1 + 1;
                dataX0{1,t1} = b;
                dataX0{2,t1} = userid;
                dataX0{3,t1} = i;
            elseif (userid == 'C') || (userid =='F')  
                t2 = t2 + 1;
                dataX1{1,t2} = b;
                dataX1{2,t2} = userid;
                dataX1{3,t2} = i;
            elseif (userid == 'D') || (userid =='E') 
                t3 = t3 + 1;
                dataX2{1,t3} = b;     
                dataX2{2,t3} = userid;
                dataX2{3,t3} = i;
            end
            
            k = k + step;
            num_trial = num_trial + 1;
        end
    end
end

dataX{1,1} =  dataX0;
dataX{1,2} =  dataX1;
dataX{1,3} =  dataX2;

%% Save data
save(strcat('./data/augment_data/',skill,'_dataX_',num2str(segment_size),'_', num2str(step),'.mat'), 'dataX'); 
