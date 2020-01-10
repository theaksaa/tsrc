for j=0:42
    disp(j);

    filename = 'D:\Traffic signs recognition and classification\Train\';
    outDirectory = 'D:\Traffic signs recognition and classification\Train_resized\';

    filename = sprintf('%s%d\\', filename, j);
    outDirectory = sprintf('%s%d\\', outDirectory, j);

    Images = dir(sprintf('%s*.png', filename));
    mkdir(outDirectory);

    for i=1:length(Images)
        ImgName = strcat(filename, Images(i).name);
        grayImage = imread(ImgName);
        newImage = imresize(grayImage,[32 32], 'bilinear');
        imwrite(newImage, strcat(outDirectory,Images(i).name));
    end

end