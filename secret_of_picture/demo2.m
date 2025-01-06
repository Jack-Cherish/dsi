% 读取普通图片和二维码图片
imgPutong = imread('test_3.jpg');
imgBarcode = imread('erweima.jpg');

% 确保两张图片的分辨率相同
[rows1, cols1, ~] = size(imgPutong);
[rows2, cols2, ~] = size(imgBarcode);

if rows1 ~= rows2 || cols1 ~= cols2
    error('两张图片的分辨率必须相同');
end

% 创建一个与普通图片分辨率相同的新的RGBA图片
imgMix = zeros(rows1, cols1, 3, 'uint8');  % 存储RGB值
alphaChannel = zeros(rows1, cols1, 'uint8');  % 存储Alpha通道

% 遍历每个像素
for w = 1:cols1
    for h = 1:rows1
        pxlPutong = imgPutong(h, w, :);
        pxlBarcode = imgBarcode(h, w, :);

        if pxlBarcode(1) > 200
            % 如果二维码上的像素为白色，则复制普通图对应位置的像素，透明度设为255（不透明）
            imgMix(h, w, 1:3) = pxlPutong(1:3);
            alphaChannel(h, w) = 255;
        else
            % 如果二维码上的像素为黑色，计算新的RGB值并设置透明度为150（半透明）
            alpha = 150;
            imgMix(h, w, 1) = uint8((double(pxlPutong(1)) - (255 - alpha)) / alpha * 255);
            imgMix(h, w, 2) = uint8((double(pxlPutong(2)) - (255 - alpha)) / alpha * 255);
            imgMix(h, w, 3) = uint8((double(pxlPutong(3)) - (255 - alpha)) / alpha * 255);
            alphaChannel(h, w) = alpha;
        end
    end
end

% 保存结果为PNG格式，分别提供RGB数据和Alpha通道
imwrite(imgMix, '图片里藏二维码.png', 'Alpha', alphaChannel);

disp('生成完毕');
