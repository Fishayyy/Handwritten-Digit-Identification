folder = 'D:\PatternRecognition\processed_images\';
filepattern = fullfile(folder,'4_*.tif');
files = dir(filepattern);
ouput_folder = 'D:\PatternRecognition\Data\output\';


files = dir('D:\PatternRecognition\Data\test\0\*.tif');
for i = 1:length(files)
    file = files(i);
    file_name = file.name;
    A = imread(strcat(strcat(file.folder,'\'),file_name));
   % A = im2gray(A);
    B = gaussf(A,120,'best');
    B = B - A;
    [B,thres] = threshold(B,'otsu',Inf);
    C = label(B,Inf,0,0);
    data = measure(C,A,'size',[],Inf,0,0);
    max = 0;
    for k = 1:length(data)
        if max < data.size(k)
            max = data.size(k);
        end
    end
    D = msr2obj(C,data,'Size',1);
    [D,thres] = threshold(D,'fixed', (3 * max) / 4);
    %[D,thres] = threshold(D,'fixed',max);
    
%     D = bskeleton(C,1,'natural');
%     E = getsinglepixel(D);
%     A = bpropagation(~E,C,Inf,-1,1);
%     A = bdilation(A,3,-1,0);
%     E = label(D,Inf,0,0);
%     data = measure(E,A,'size',[],Inf,0,0);
%     if length(data.size) > 1
%         min = 1000000;
%         for k = 1:length(data)
%             if min > data.size(k)
%                 min = data.size(k);
%             end
%         end
%         D = msr2obj(E,data,'Size',1);
%         [D,thres] = threshold(D,'isodata',min);
%     end
    
    
    file_name = strcat(ouput_folder,file_name);
    writeim(D,file_name,'TIFF',0,[])

    
end