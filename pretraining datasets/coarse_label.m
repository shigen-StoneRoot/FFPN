img_dir = 'ADAM\image\';
hs_tar = 'ADAM\hessian\';
tar = 'ADAM\coarse_gt\';

filenames = ls(img_dir);
filenames = cellstr(filenames);

len = length(filenames);
for i=3:len
    sub_path = strcat(img_dir, filenames{i});
    sub_nii = load_untouch_nii(sub_path);
    img = double(sub_nii.img);
    img = img / max(img(:));
  
    fm = fibermetric(img, 'StructureSensitivity', 0.01);

    hs_nii = sub_nii;
    hs_nii.img = fm;
    save_untouch_nii(hs_nii, strcat(hs_tar, filenames{i}));
    
    thres = quantile(fm(fm~=0), 0.97);
    mask = fm > thres;
    
    imLabel = bwlabeln(mask);
    stats = regionprops(imLabel, 'Area');
    [b, index] = sort([stats.Area], 'descend');
    bw = ismember(imLabel, index(1:5));

    mask = bw + 0;
    
    new_nii = sub_nii;
    new_nii.img = mask;
    
    save_untouch_nii(new_nii, strcat(tar, filenames{i}));
    
    display(i);
    
end
