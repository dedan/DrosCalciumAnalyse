
inpath = '/Users/dedan/projects/fu/results/alignment/images/LIN_111108a_raw/';
outpath = '/Users/dedan/projects/fu/results/alignment/images/LIN_111108a_myalign_all/';
transform = 'affine'; 
NoI = 5; % number of iterations
NoL = 1;  % number of pyramid-levels
init=[eye(2) zeros(2,1)];

% for i = 0:99
% 
%     disp(['loop: ' num2str(i)])
%     if mod(i, 40) == 0
%         infile1 = [inpath sprintf('test%02d.pgm', i)];
%         im1 = imread(infile1);    
%         outfile = [outpath sprintf('out%02d.pgm', i)];
%         imwrite(uint8(im1), outfile);
%     else
%         infile1 = [outpath sprintf('out%02d.pgm', i)];
%         im1 = imread(infile1);    
%     end
%     infile2 = [inpath sprintf('test%02d.pgm', i+1)];
%         
%     im2 = imread(infile2);
% 
%     [results, final_warp, warped_image] = ecc(im2, im1, NoL, NoI, transform, init);
%     outfile = [outpath sprintf('out%02d', i+1) '.pgm'];
%     imwrite(uint8(warped_image), outfile);
%     
% end
% disp('finished')



for i = 0:299

    disp(['loop: ' num2str(i)])
    infile1 = [inpath sprintf('test%03d.pgm', i)];
    infile2 = [inpath sprintf('test%03d.pgm', i+1)];
    im2 = imread(infile2);

    [results, final_warp, warped_image] = ecc(im2, im1, NoL, NoI, transform, init);
    outfile = [outpath sprintf('out%03d', i+1) '.pgm'];
    imwrite(uint8(warped_image), outfile);
    
end
disp('finished')