function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------

% step up parameters 
opts.imageSize = [256, 256] ;
opts.border = [0, 0] ; 
opts.keepAspect = true ;
opts.numAugments = 1 ;
opts.transformation = 'none'; 
opts.affine = false;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
%%opts = vl_argparse(opts, varargin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The input is original image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 load('multi.mat');
 sample_path = strcat(imdb.coverDir, num2str(batch(1)), '.pgm');
 cover = imread(sample_path);
 p_size = 20;
 im = single(zeros(size(cover,1),size(cover,2),1,2*p_size));
 
 %predefined cover images and stego images
 predefined_cover = imdb.meta.stable_cover;
 predefined_stego = imdb.meta.stable_stego;
 
 for i = 1 : p_size-1
   cover_path = strcat(predefined_cover, num2str(i+(multi-1)*p_size), '.pgm'); 
   stego_path = strcat(predefined_stego, num2str(i+(multi-1)*p_size), '.pgm');
   cover = imread(cover_path);
   stego = imread(stego_path);  
   im(:, :, 1, 2*i-1) = single(cover);
   im(:, :, 1, 2*i) = single(stego);
 end

 test_cover_path = strcat(imdb.coverDir, num2str(batch(1)), '.pgm'); 
 test_stego_path = strcat(imdb.stegoDir, num2str(batch(1)), '.pgm');
 cover = imread(test_cover_path);
 stego = imread(test_stego_path);  
 im(:, :, 1, 2*p_size-1) = single(cover);
 im(:, :, 1, 2*p_size) = single(stego);
 
 
labels = ones(1,2*(p_size-1)) + (sign((-1).^(1:2*(p_size-1)))+1)/2;
labels(end+1) = 1;
labels(end+1) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prefix = 'predictions/';
if ~exist(strcat(prefix,num2str(multi)))
  mkdir(strcat(prefix,num2str(multi)));
  mkdir(strcat(prefix,num2str(multi),'/predicted_labels'));
end
fprintf('the %d-th image, ', batch);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargout > 0
    if useGpu
        im = gpuArray(im) ;
    end
    inputs = {'data', im, 'label', labels} ;
end

end