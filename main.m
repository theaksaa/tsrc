load rcnn;

testImage = imread('Test/3.jpg');

[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128)

[score, idx] = max(score);
bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %0.2f%%)', label(idx), score*100);

outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation, 'LineWidth', 3, 'TextBoxOpacity',0.9,'FontSize',18);

figure
imshow(outputImage)