# Credit Card Number Reader

I implemented a credit card reader using Tensorflow with four main steps:
1. Creating a Credit Card Number Dataset
2. Data Augmentation and model training
3. Extracting a card number section from card picture
4. Identify digits using the model

Unfortunately, there isn't an official standard credit card number font - some of the fonts go by the names Farrington 7B, OCR-B, SecurePay, OCR-A and MICR E13B. However, there seem to be two main font variations used in credit cards: [this](data/creditcard_digits1.jpg) and [this](data/creditcard_digits2.jpg).
So, I trained my model for both of them.

First I create dictionaries for dataset: data/credit_card/train/ and data/credit_card/test/ (separated folders for each digits).
Then I implement five data augmentation functions. What I'm doing here is taking two samples of each digit said above and adding small variations to each of them. This is very similar to Keras's Data Augmentation, however, I'm using OpenCV to create an augmented dataset instead. I also use Keras later to augment even further. The functions are:
1. add_noise() - This function introduces some noise elements to the image
2. pixelate() - This function resizes the image then upscales/upsamples it. This degrades the quality and is meant to simulate blur to the image from either a shakey or poor quality camera.
3. stretch() - This simulates some variation in resizing where it stretches the image to a small random size
4. pre_process() - This is a simple function that applies OTSU Binarization to the image and resizes it. I use this on the extracted digits to create a clean dataset akin to the MNIST style format.
5. digit_augmentation() - This one simply uses the other image manipulation functions (calls them randomly).

[This is how our augmented fonts look like](data/augmented_fonts.jpg).

Then I create a dataset. After splitting I've got:
for training, 1000 images for each digit. As long as we have two types of font it hits 20k images total.
for testing, 200 images for each digit so it's 4k images total.

Then I take advantage of Keras's Data Augmentation and apply some small rotations, shifts, shearing and zooming. Finally, I train my model for 5 EPOCHS.

We now need to extract the credit card number region. In the code below, I'm always resizing the extracted credit card image to a size of 640 x 403. The odd choice of 403 was chosen because 640:403 is the actual ratio of a credit card. I'm trying to maintain dimensions as accurately as possible so that we don't necessarily warp the image too much.
We're resizing all extracted digits to 32 x 32, but even still keeping the initial ratio correct will only help our classifier accuracy.
As such, because of the fixed size, we can now extract the region ([(55, 210), (640, 290)]) easily from the image.

Steps:
1. Load the image.
2. Use Canny Edge detection to identify the edges of the card
3. Use cv2.findContours() to extract the largest contour (which I assume will be the credit card)
4. Use the function four_point_transform() and order_points() to adjust the perspective of the card. It creates a top-down type of view that is useful because:
. It standardizes the view of the card so that the credit card digits are always roughly in the same area.
. It removes/reduces skew and warped perspectives from the image when taken from a camera. All cameras unless taken exactly top-down will introduce some skew to text/digits. This is why scanners always produce more realistic looking images.

[This is how extracted image looks like](data/credit_card_extracted_digits.jpg).


Finally, for each test credit card, I:
1. Load grayscale extracted image
2. Apply the Canny Edge algorithm
3. Use findCountours to isolate the digits
4. Sort the contours by size (so that smaller irrelevant contours aren't used)
5. Sort it left to right by creating a function that returns the x-coordinate of a contour
6. Once I have cleaned up contours, I find the bounding rectangle of the contour which gives me an enclosed rectangle around the digit. (To ensure these contours are valid I do extract only contours meeting the minimum width and height expectations).
7. Then I take each extracted digit, use my pre_processing function (which applies OTSU Binarization and resizes it), then breakdown that image array so that it can be loaded into the classifier.

P.S. I tried my best to get as many card photos as possible, I managed to collect about 25 cards and the model identified all of them with 100% correctness. 
