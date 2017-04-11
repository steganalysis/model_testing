function imdb= cnn_steganalysis_setup_data(file_add, Num, accomp_cover, accomp_stego)
   % this function is to read input images
   imdb.coverDir = file_add;
   imdb.stegoDir = file_add;
    
  % descriptions to the image database
  imdb.meta.sets = {'train', 'val'};
  imdb.meta.classes = {1,2};
  imdb.meta.stable_cover = accomp_cover;
  imdb.meta.stable_stego = accomp_stego;
  
  % details to the image database
  set = [2*ones(1, Num)]; % 2 represents the image is for validation
  imdb.images.set = set;
end