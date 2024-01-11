%% Script to load, preprocess and save blood MNIST images to .amat format
close all
clear all
clc

disp('Hello!')

%% Cancer cells

myDir = 'K:\PhData\bloodmnist\train\0';
myFiles = dir(fullfile(myDir,'*.png')); %gets all png files in struct

total_length = 11959;
imglinearlength = 785;
final_file = zeros(total_length, imglinearlength);
disp('finished allocating file');
length_0 = length(myFiles);

for i=1:length_0
    
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    [A, Fs] = imread(fullFileName);
    
    %% Scale to [0,1]
    I = rescale(A);
    
    %% Convert to greyscale
    I = rgb2gray(I);
    
    %% Augment
    
    Img_rot1 = imrotate(I,90,'bilinear','crop');
    
    Img_rot2 = imrotate(I,180,'bilinear','crop');
    
    Img_rot3 = imrotate(I,270,'bilinear','crop');
    
    Img_flip1 = fliplr(I);
    
    Img_flip2 = imrotate(Img_flip1,90,'bilinear','crop');
    
    Img_flip3 = imrotate(Img_flip1,180,'bilinear','crop');
    
    Img_flip4 = imrotate(Img_flip1,270,'bilinear','crop');
    
    %% Reformat. They need to be stored row-wise, with the label at the end.
    
    % % Reformat row-wise. That's why we transpose.
    I_reformat = reshape(I',1,imglinearlength-1);
    I_reformat_rot1 = reshape(Img_rot1',1,imglinearlength-1);
    I_reformat_rot2 = reshape(Img_rot2',1,imglinearlength-1);
    I_reformat_rot3 = reshape(Img_rot3',1,imglinearlength-1);
    I_reformat_flip1 = reshape(Img_flip1',1,imglinearlength-1);
    I_reformat_flip2 = reshape(Img_flip2',1,imglinearlength-1);
    I_reformat_flip3 = reshape(Img_flip3',1,imglinearlength-1);
    I_reformat_flip4 = reshape(Img_flip4',1,imglinearlength-1);
    
    %% Append label
    I_reformat(1,imglinearlength)=0;
    I_reformat_rot1(1,imglinearlength)=0;
    I_reformat_rot2(1,imglinearlength)=0;
    I_reformat_rot3(1,imglinearlength)=0;
    I_reformat_flip1(1,imglinearlength)=0;
    I_reformat_flip2(1,imglinearlength)=0;
    I_reformat_flip3(1,imglinearlength)=0;
    I_reformat_flip4(1,imglinearlength)=0;
    
    %% Insert into big matrix
    final_file(1+(i-1)*8,:)=I_reformat; % The eight is because we use eight augmentations
    final_file(2+(i-1)*8,:)=I_reformat_rot1;
    final_file(3+(i-1)*8,:)=I_reformat_rot2;
    final_file(4+(i-1)*8,:)=I_reformat_rot3;
    final_file(5+(i-1)*8,:)=I_reformat_flip1; 
    final_file(6+(i-1)*8,:)=I_reformat_flip2;
    final_file(7+(i-1)*8,:)=I_reformat_flip3;
    final_file(8+(i-1)*8,:)=I_reformat_flip4;
    
end

%% Class 1

myDir = 'K:\PhData\bloodmnist\train\1';
myFiles = dir(fullfile(myDir,'*.png')); %gets all png files in struct

length_1 = length(myFiles);

for i=1:length_1
    
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    [A, Fs] = imread(fullFileName);
    
    %% Scale to [0,1]
    I = rescale(A);
    
    %% Convert to greyscale
    I = rgb2gray(I);
    
    %% Reformat. They need to be stored row-wise, with the label at the end.
    
    % % Reformat row-wise. That's why we transpose.
    I_reformat = reshape(I',1,imglinearlength-1);
    I_reformat_rot1 = reshape(Img_rot1',1,imglinearlength-1);
    I_reformat_rot2 = reshape(Img_rot2',1,imglinearlength-1);
    I_reformat_rot3 = reshape(Img_rot3',1,imglinearlength-1);
    I_reformat_flip1 = reshape(Img_flip1',1,imglinearlength-1);
    I_reformat_flip2 = reshape(Img_flip2',1,imglinearlength-1);
    I_reformat_flip3 = reshape(Img_flip3',1,imglinearlength-1);
    I_reformat_flip4 = reshape(Img_flip4',1,imglinearlength-1);
    
    %% Append label
    I_reformat(1,imglinearlength)=1;
    I_reformat_rot1(1,imglinearlength)=1;
    I_reformat_rot2(1,imglinearlength)=1;
    I_reformat_rot3(1,imglinearlength)=1;
    I_reformat_flip1(1,imglinearlength)=1;
    I_reformat_flip2(1,imglinearlength)=1;
    I_reformat_flip3(1,imglinearlength)=1;
    I_reformat_flip4(1,imglinearlength)=1;
    
    %% Insert into big matrix
    final_file(1+(i-1)*8+length_0*8,:)=I_reformat; % The eight is because we use eight augmentations
    final_file(2+(i-1)*8+length_0*8,:)=I_reformat_rot1;
    final_file(3+(i-1)*8+length_0*8,:)=I_reformat_rot2;
    final_file(4+(i-1)*8+length_0*8,:)=I_reformat_rot3;
    final_file(5+(i-1)*8+length_0*8,:)=I_reformat_flip1; 
    final_file(6+(i-1)*8+length_0*8,:)=I_reformat_flip2;
    final_file(7+(i-1)*8+length_0*8,:)=I_reformat_flip3;
    final_file(8+(i-1)*8+length_0*8,:)=I_reformat_flip4;
    
end

myDir = 'K:\PhData\bloodmnist\train\2';
myFiles = dir(fullfile(myDir,'*.png')); %gets all png files in struct

length_2 = length(myFiles);

for i=1:length_2
    
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    [A, Fs] = imread(fullFileName);
    
    %% Scale to [0,1]
    I = rescale(A);
    
    %% Convert to greyscale
    I = rgb2gray(I);
    
    %% Reformat. They need to be stored row-wise, with the label at the end.
    
    % % Reformat row-wise. That's why we transpose.
    I_reformat = reshape(I',1,imglinearlength-1);
    I_reformat_rot1 = reshape(Img_rot1',1,imglinearlength-1);
    I_reformat_rot2 = reshape(Img_rot2',1,imglinearlength-1);
    I_reformat_rot3 = reshape(Img_rot3',1,imglinearlength-1);
    I_reformat_flip1 = reshape(Img_flip1',1,imglinearlength-1);
    I_reformat_flip2 = reshape(Img_flip2',1,imglinearlength-1);
    I_reformat_flip3 = reshape(Img_flip3',1,imglinearlength-1);
    I_reformat_flip4 = reshape(Img_flip4',1,imglinearlength-1);
    
    %% Append label
    I_reformat(1,imglinearlength)=2;
    I_reformat_rot1(1,imglinearlength)=2;
    I_reformat_rot2(1,imglinearlength)=2;
    I_reformat_rot3(1,imglinearlength)=2;
    I_reformat_flip1(1,imglinearlength)=2;
    I_reformat_flip2(1,imglinearlength)=2;
    I_reformat_flip3(1,imglinearlength)=2;
    I_reformat_flip4(1,imglinearlength)=2;
    
    %% Insert into big matrix
    final_file(1+(i-1)*8+length_0*8+length_1*8,:)=I_reformat; % The eight is because we use eight augmentations
    final_file(2+(i-1)*8+length_0*8+length_1*8,:)=I_reformat_rot1;
    final_file(3+(i-1)*8+length_0*8+length_1*8,:)=I_reformat_rot2;
    final_file(4+(i-1)*8+length_0*8+length_1*8,:)=I_reformat_rot3;
    final_file(5+(i-1)*8+length_0*8+length_1*8,:)=I_reformat_flip1; 
    final_file(6+(i-1)*8+length_0*8+length_1*8,:)=I_reformat_flip2;
    final_file(7+(i-1)*8+length_0*8+length_1*8,:)=I_reformat_flip3;
    final_file(8+(i-1)*8+length_0*8+length_1*8,:)=I_reformat_flip4;
    
end

myDir = 'K:\PhData\bloodmnist\train\3';
myFiles = dir(fullfile(myDir,'*.png')); %gets all png files in struct

length_3 = length(myFiles);

for i=1:length_3
    %% Load images
    
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    [A, Fs] = imread(fullFileName);
    
    %% Scale to [0,1]
    I = rescale(A);
    
    %% Convert to greyscale
    I = rgb2gray(I);
    
    %% Reformat. They need to be stored row-wise, with the label at the end.
    
    % % Reformat row-wise. That's why we transpose.
    I_reformat = reshape(I',1,imglinearlength-1);
    I_reformat_rot1 = reshape(Img_rot1',1,imglinearlength-1);
    I_reformat_rot2 = reshape(Img_rot2',1,imglinearlength-1);
    I_reformat_rot3 = reshape(Img_rot3',1,imglinearlength-1);
    I_reformat_flip1 = reshape(Img_flip1',1,imglinearlength-1);
    I_reformat_flip2 = reshape(Img_flip2',1,imglinearlength-1);
    I_reformat_flip3 = reshape(Img_flip3',1,imglinearlength-1);
    I_reformat_flip4 = reshape(Img_flip4',1,imglinearlength-1);
    
    %% Append label
    I_reformat(1,imglinearlength)=3;
    I_reformat_rot1(1,imglinearlength)=3;
    I_reformat_rot2(1,imglinearlength)=3;
    I_reformat_rot3(1,imglinearlength)=3;
    I_reformat_flip1(1,imglinearlength)=3;
    I_reformat_flip2(1,imglinearlength)=3;
    I_reformat_flip3(1,imglinearlength)=3;
    I_reformat_flip4(1,imglinearlength)=3;
    
    %% Insert into big matrix
    final_file(1+(i-1)*8+length_0*8+length_1*8+length_2*8,:)=I_reformat; % The eight is because we use eight augmentations
    final_file(2+(i-1)*8+length_0*8+length_1*8+length_2*8,:)=I_reformat_rot1;
    final_file(3+(i-1)*8+length_0*8+length_1*8+length_2*8,:)=I_reformat_rot2;
    final_file(4+(i-1)*8+length_0*8+length_1*8+length_2*8,:)=I_reformat_rot3;
    final_file(5+(i-1)*8+length_0*8+length_1*8+length_2*8,:)=I_reformat_flip1; 
    final_file(6+(i-1)*8+length_0*8+length_1*8+length_2*8,:)=I_reformat_flip2;
    final_file(7+(i-1)*8+length_0*8+length_1*8+length_2*8,:)=I_reformat_flip3;
    final_file(8+(i-1)*8+length_0*8+length_1*8+length_2*8,:)=I_reformat_flip4;
    
end

myDir = 'K:\PhData\bloodmnist\train\4';
myFiles = dir(fullfile(myDir,'*.png')); %gets all png files in struct

length_4 = length(myFiles);

for i=1:length_4
    %% Load images
    
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    [A, Fs] = imread(fullFileName);
    
    %% Scale to [0,1]
    I = rescale(A);
    
    %% Convert to greyscale
    I = rgb2gray(I);
    
    %% Reformat. They need to be stored row-wise, with the label at the end.
    
    % % Reformat row-wise. That's why we transpose.
    I_reformat = reshape(I',1,imglinearlength-1);
    I_reformat_rot1 = reshape(Img_rot1',1,imglinearlength-1);
    I_reformat_rot2 = reshape(Img_rot2',1,imglinearlength-1);
    I_reformat_rot3 = reshape(Img_rot3',1,imglinearlength-1);
    I_reformat_flip1 = reshape(Img_flip1',1,imglinearlength-1);
    I_reformat_flip2 = reshape(Img_flip2',1,imglinearlength-1);
    I_reformat_flip3 = reshape(Img_flip3',1,imglinearlength-1);
    I_reformat_flip4 = reshape(Img_flip4',1,imglinearlength-1);
    
    %% Append label
    I_reformat(1,imglinearlength)=4;
    I_reformat_rot1(1,imglinearlength)=4;
    I_reformat_rot2(1,imglinearlength)=4;
    I_reformat_rot3(1,imglinearlength)=4;
    I_reformat_flip1(1,imglinearlength)=4;
    I_reformat_flip2(1,imglinearlength)=4;
    I_reformat_flip3(1,imglinearlength)=4;
    I_reformat_flip4(1,imglinearlength)=4;
    
    %% Insert into big matrix
    final_file(1+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8,:)=I_reformat; % The eight is because we use eight augmentations
    final_file(2+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8,:)=I_reformat_rot1;
    final_file(3+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8,:)=I_reformat_rot2;
    final_file(4+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8,:)=I_reformat_rot3;
    final_file(5+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8,:)=I_reformat_flip1; 
    final_file(6+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8,:)=I_reformat_flip2;
    final_file(7+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8,:)=I_reformat_flip3;
    final_file(8+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8,:)=I_reformat_flip4;
    
end

myDir = 'K:\PhData\bloodmnist\train\5';
myFiles = dir(fullfile(myDir,'*.png')); %gets all png files in struct

length_5 = length(myFiles);

for i=1:length_5
    %% Load images
    
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    [A, Fs] = imread(fullFileName);
    
    %% Scale to [0,1]
    I = rescale(A);
    
    %% Convert to greyscale
    I = rgb2gray(I);
    
    %% Reformat. They need to be stored row-wise, with the label at the end.
    
    % % Reformat row-wise. That's why we transpose.
    I_reformat = reshape(I',1,imglinearlength-1);
    I_reformat_rot1 = reshape(Img_rot1',1,imglinearlength-1);
    I_reformat_rot2 = reshape(Img_rot2',1,imglinearlength-1);
    I_reformat_rot3 = reshape(Img_rot3',1,imglinearlength-1);
    I_reformat_flip1 = reshape(Img_flip1',1,imglinearlength-1);
    I_reformat_flip2 = reshape(Img_flip2',1,imglinearlength-1);
    I_reformat_flip3 = reshape(Img_flip3',1,imglinearlength-1);
    I_reformat_flip4 = reshape(Img_flip4',1,imglinearlength-1);
    
    %% Append label
    I_reformat(1,imglinearlength)=5;
    I_reformat_rot1(1,imglinearlength)=5;
    I_reformat_rot2(1,imglinearlength)=5;
    I_reformat_rot3(1,imglinearlength)=5;
    I_reformat_flip1(1,imglinearlength)=5;
    I_reformat_flip2(1,imglinearlength)=5;
    I_reformat_flip3(1,imglinearlength)=5;
    I_reformat_flip4(1,imglinearlength)=5;
    
    %% Insert into big matrix
    final_file(1+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8,:)=I_reformat; % The eight is because we use eight augmentations
    final_file(2+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8,:)=I_reformat_rot1;
    final_file(3+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8,:)=I_reformat_rot2;
    final_file(4+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8,:)=I_reformat_rot3;
    final_file(5+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8,:)=I_reformat_flip1; 
    final_file(6+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8,:)=I_reformat_flip2;
    final_file(7+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8,:)=I_reformat_flip3;
    final_file(8+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8,:)=I_reformat_flip4;
    
end

myDir = 'K:\PhData\bloodmnist\train\6';
myFiles = dir(fullfile(myDir,'*.png')); %gets all png files in struct

length_6 = length(myFiles);

for i=1:length_6
    %% Load images
    
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    [A, Fs] = imread(fullFileName);
    
    %% Scale to [0,1]
    I = rescale(A);
    
    %% Convert to greyscale
    I = rgb2gray(I);
    %imtool(Img);
    
    %% Reformat. They need to be stored row-wise, with the label at the end.
    
    % % Reformat row-wise. That's why we transpose.
    I_reformat = reshape(I',1,imglinearlength-1);
    I_reformat_rot1 = reshape(Img_rot1',1,imglinearlength-1);
    I_reformat_rot2 = reshape(Img_rot2',1,imglinearlength-1);
    I_reformat_rot3 = reshape(Img_rot3',1,imglinearlength-1);
    I_reformat_flip1 = reshape(Img_flip1',1,imglinearlength-1);
    I_reformat_flip2 = reshape(Img_flip2',1,imglinearlength-1);
    I_reformat_flip3 = reshape(Img_flip3',1,imglinearlength-1);
    I_reformat_flip4 = reshape(Img_flip4',1,imglinearlength-1);
    
    %% Append label
    I_reformat(1,imglinearlength)=6;
    I_reformat_rot1(1,imglinearlength)=6;
    I_reformat_rot2(1,imglinearlength)=6;
    I_reformat_rot3(1,imglinearlength)=6;
    I_reformat_flip1(1,imglinearlength)=6;
    I_reformat_flip2(1,imglinearlength)=6;
    I_reformat_flip3(1,imglinearlength)=6;
    I_reformat_flip4(1,imglinearlength)=6;
    
    %% Insert into big matrix
    final_file(1+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8,:)=I_reformat; % The eight is because we use eight augmentations
    final_file(2+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8,:)=I_reformat_rot1;
    final_file(3+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8,:)=I_reformat_rot2;
    final_file(4+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8,:)=I_reformat_rot3;
    final_file(5+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8,:)=I_reformat_flip1; 
    final_file(6+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8,:)=I_reformat_flip2;
    final_file(7+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8,:)=I_reformat_flip3;
    final_file(8+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8,:)=I_reformat_flip4;
    
end

myDir = 'K:\PhData\bloodmnist\train\7';
myFiles = dir(fullfile(myDir,'*.png')); %gets all png files in struct

length_7 = length(myFiles);

for i=1:length_7
    %% Load images
    
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    [A, Fs] = imread(fullFileName);
    
    %% Scale to [0,1]
    I = rescale(A);
    
    %% Convert to greyscale
    I = rgb2gray(I);
    
    %% Reformat. They need to be stored row-wise, with the label at the end.
    
    % % Reformat row-wise. That's why we transpose.
    I_reformat = reshape(I',1,imglinearlength-1);
    I_reformat_rot1 = reshape(Img_rot1',1,imglinearlength-1);
    I_reformat_rot2 = reshape(Img_rot2',1,imglinearlength-1);
    I_reformat_rot3 = reshape(Img_rot3',1,imglinearlength-1);
    I_reformat_flip1 = reshape(Img_flip1',1,imglinearlength-1);
    I_reformat_flip2 = reshape(Img_flip2',1,imglinearlength-1);
    I_reformat_flip3 = reshape(Img_flip3',1,imglinearlength-1);
    I_reformat_flip4 = reshape(Img_flip4',1,imglinearlength-1);
    
    %% Append label
    I_reformat(1,imglinearlength)=7;
    I_reformat_rot1(1,imglinearlength)=7;
    I_reformat_rot2(1,imglinearlength)=7;
    I_reformat_rot3(1,imglinearlength)=7;
    I_reformat_flip1(1,imglinearlength)=7;
    I_reformat_flip2(1,imglinearlength)=7;
    I_reformat_flip3(1,imglinearlength)=7;
    I_reformat_flip4(1,imglinearlength)=7;
    
    %% Insert into big matrix
    final_file(1+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8+length_6*8,:)=I_reformat; % The eight is because we use eight augmentations
    final_file(2+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8+length_6*8,:)=I_reformat_rot1;
    final_file(3+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8+length_6*8,:)=I_reformat_rot2;
    final_file(4+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8+length_6*8,:)=I_reformat_rot3;
    final_file(5+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8+length_6*8,:)=I_reformat_flip1; 
    final_file(6+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8+length_6*8,:)=I_reformat_flip2;
    final_file(7+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8+length_6*8,:)=I_reformat_flip3;
    final_file(8+(i-1)*8+length_0*8+length_1*8+length_2*8+length_3*8+length_4*8+length_5*8+length_6*8,:)=I_reformat_flip4;
    
end

dlmwrite('Cells_train_aug.txt', final_file, 'precision', 7, 'delimiter', ' ');
disp('finished writing file');