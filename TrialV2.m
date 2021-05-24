currentFolder = pwd;
rootfolder_path = strcat(currentFolder,'\processed_images\');
rootfolder = dir(rootfolder_path);
output_folder = "";

for p = 3:length(rootfolder)
    sequence_folders = rootfolder(p);
    folder_path = strcat(sequence_folders.folder,'\');
    folder_path = strcat(folder_path,sequence_folders.name);
    folders = dir(folder_path);
    grey = "";
    for i = 1:length(folders)
        if(folders(i).name == "grayscale")
            grey = folders(i);
            output_name = strcat(grey.folder,'\');
            output_name = strcat(output_name,grey.name);
            output_folder = replace(output_name,'grayscale','binary\');
            if ~exist(output_folder, 'dir')
                mkdir(output_folder)
            end
        end
    end
    if ~(isempty(grey))
        folder_path = strcat(grey.folder,'\');
        folder_path = strcat(folder_path,grey.name);
        filepattern = fullfile(folder_path,'*.tif');
        files = dir(filepattern);
        for i = 1:length(files)
            file = files(i);
            file_name = file.name;
            file_name = strcat(strcat(file.folder,'\'),file_name);
            A = imread(file_name);
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
            output_name = replace(file_name,'grayscale','binary');
            writeim(D,output_name,'TIFF',0,[])
        end
    end 
    
end