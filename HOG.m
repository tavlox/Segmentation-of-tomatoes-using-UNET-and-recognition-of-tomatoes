function [D_pravi_ucenje,D_pravi_test,D_pravi_param, imPath_test] = HOG(path_img_train,path_img_test, class)
    full_img_file_train = dir(fullfile(path_img_train,'/*.png'));
    file_names_train = {full_img_file_train.name};
  
   for bb = 1:length(file_names_train)
        imPath_train{bb} = fullfile(path_img_train,file_names_train{bb});
        image_read_train{bb} = imread(imPath_train{bb});
        if size(image_read_train{bb},3) == 3
            image_read_train{bb} = im2single(rgb2gray(image_read_train{bb}));
            image_read_train{bb} = imresize(image_read_train{bb},[64 128]); % resize na 64x128
            image_read_train{bb} = histeq(image_read_train{bb});

        else
            image_read_train{bb} = im2single(image_read_train{bb});
            image_read_train{bb} = imresize(image_read_train{bb},[64 128]); % resize na 64x128
            image_read_train{bb} = histeq(image_read_train{bb});


        end
    end
    
    
   full_img_file_test = dir(fullfile(path_img_test,'/*.png'));
   file_names_test = {full_img_file_test.name};

   for bb = 1:length(file_names_test)
        imPath_test{bb} = fullfile(path_img_test,file_names_test{bb});
        image_read_test{bb} = imread(imPath_test{bb});
       
        if size(image_read_test{bb},3) == 3
            image_read_test{bb} = im2single(rgb2gray(image_read_test{bb}));
            image_read_test{bb} = imresize(image_read_test{bb},[64 128]); % resize na 64x128
            image_read_test{bb} = histeq(image_read_test{bb});


        else
            image_read_test{bb} = im2single(image_read_test{bb});
            image_read_test{bb} = imresize(image_read_test{bb},[64 128]); % resize na 64x128
            image_read_test{bb} = histeq(image_read_test{bb});


        end
        cropped_testiranje = image_read_test{bb};
        d_test = vl_hog(cropped_testiranje,8,'numOrientations',12,'variant', 'dalaltriggs');
        d_size_test = size(d_test);
        jpg_files_name_test = full_img_file_test(bb).name;
        %inicializiramo strukturo, ki vsebuje ime razreda in deskriptorje za vsako
        %sliko:
        if bb == 1
            D_test = repmat(struct('name',"img",'descriptors',zeros(d_size_test(1),d_size_test(2))), length(file_names_test), 1 );
        end

        D_test(bb).name = class;
        D_test(bb).descriptors = d_test;
        
    end
    D_pravi_test= D_test;
    D_pravi_test = D_pravi_test';
        
    for dg = 1:floor(length(file_names_train)/2)
        cropped_ucenje_dejansko = image_read_train{dg};
        d_ucenje = vl_hog(cropped_ucenje_dejansko,8,'numOrientations',12,'variant', 'dalaltriggs');
        D_size_ucenje = size(d_ucenje);
        jpg_files_name_dejansko = full_img_file_train(dg).name;
        %inicializiramo strukturo, ki vsebuje ime razreda in deskriptorje za vsako
        %sliko:
        if dg == 1
            D = repmat(struct('name',"img",'descriptors',zeros(D_size_ucenje(1),D_size_ucenje(2))), length(file_names_train), 1 );
        end

        D(dg).name = class;
        D(dg).descriptors = d_ucenje;
        
    end
    D_pravi_ucenje= D(1:floor(length(file_names_train)/2));

    
     
    for dt = floor(length(file_names_train)/2)+1:length(file_names_train)
        cropped_ucenje_param = image_read_train{dt};
        d_param_ucenje = vl_hog(cropped_ucenje_param,8,'numOrientations',12,'variant', 'dalaltriggs');
        D_size_param = size(dt);
        jpg_files_name_param = full_img_file_train(dt).name;
        %inicializiramo strukturo, ki vsebuje ime slike in deskriptorje za vsako
        %sliko:
        if dt == 1
            D_param = repmat(struct('name',"img",'descriptors',zeros(D_size_param(1),D_size_param(2))), length(file_names_train), 1 );
        end

        D_param(dt).name = class;
        D_param(dt).descriptors = d_param_ucenje;

    end
    D_pravi_param= D_param(floor(length(file_names_train)/2)+1:length(file_names_train));
    D_pravi_param = D_pravi_param';
    
   
    
end