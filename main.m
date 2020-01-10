load rcnn;

testImage = imread('Test/8.jpg');

[bboxes, score, idx] = detect(rcnn, testImage, 'MiniBatchSize', 128)

% figure
% hold on;
% imshow(testImage);

annotation = [];
bbox = [];

bbox = [];
for i = 1:size(bboxes, 1)
    if score(i,1)*100 >= 0.00
        t = sprintf('%s (%0.2f%%)', idx(i,1), score(i,1)*100);
        annotation{i} = t;
        bbox = [bbox; bboxes(i,:)];
        %text(bboxes(i,1:1), bboxes(i,2:2) - 15, t, 'Fontsize', 10, 'Color', 'b','TextBoxOpacity',0.9);
        %rectangle('Position', bboxes(i,:), 'Edgecolor', 'r');
    end
end

if size(annotation, 2) ~= 0
    outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, cellstr(annotation), ...
                                         'LineWidth', 3, 'TextBoxOpacity', 0.9,'FontSize', 18, 'Color', 'yellow');
    imshow(outputImage);
else
    imshow(testImage);
end


%hold off;

