function big_with_small(big_img_path, small_img_path, res_img_path)
    % 大图里藏小图
    dst_img = generate_img(big_img_path, small_img_path);
    imwrite(dst_img, res_img_path);
end

function dst_img = generate_img(big_img_path, small_img_path)
    big_img = imread(big_img_path);
    sml_img = imread(small_img_path);

    dst_img = big_img;

    [big_h, big_w, ~] = size(big_img);
    [sml_h, sml_w, ~] = size(sml_img);

    stepx = big_w / sml_w;
    stepy = big_h / sml_h;

    for m = 0:(sml_w - 1)
        for n = 0:(sml_h - 1)
            map_col = floor(m * stepx + stepx * 0.5);
            map_row = floor(n * stepy + stepy * 0.5);

            if map_col < big_w && map_row < big_h
                dst_img(map_row + 1, map_col + 1, :) = sml_img(n + 1, m + 1, :);
            end
        end
    end
end

% 将小图藏于大图中，并保存结果
big_img_path = 'test_1.jpg';
small_img_path = 'test_2.jpg';
res_img_path = '图片里藏小图.png';
big_with_small(big_img_path, small_img_path, res_img_path);
