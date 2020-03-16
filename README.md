# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## Solution

##### MP.1 Data Buffer Optimization
```c++
deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
DataFrame frame;
frame.cameraImg = imgGray;
if (dataBuffer.size() == dataBufferSize)
    dataBuffer.pop_front();
dataBuffer.push_back(frame);
```

##### MP.2 Keypoint Detection

```c++
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis) {
    cv::Ptr<cv::Feature2D> detector;
    if (detectorType.compare("FAST") == 0) detector = cv::FastFeatureDetector::create();
    else if (detectorType.compare("BRISK") == 0) detector = cv::BRISK::create();
    else if (detectorType.compare("ORB") == 0) detector = cv::ORB::create();
    else if (detectorType.compare("AKAZE") == 0) detector = cv::AKAZE::create();
    else if (detectorType.compare("SIFT") == 0) detector = cv::xfeatures2d::SIFT::create();
    else throw invalid_argument(detectorType + " unsupported detectorType");
    detector->detect(img, keypoints);
    if (bVis) {
        // Visualize the keypoints
        string windowName = detectorType + " Keypoint Detection Results";
        cv::namedWindow(windowName);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
```

#### MP.3 Keypoint Removal

```c++
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle) {
            vector<cv::KeyPoint> filteredKeypoints;
            for (auto kp : keypoints) {
                if (vehicleRect.contains(kp.pt))
                    filteredKeypoints.push_back(kp);
            }
            keypoints = filteredKeypoints;
        }
```

#### MP.4 Keypoint Descriptors

```c++
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0) {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType.compare("BRIEF") == 0) extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    else if (descriptorType.compare("ORB") == 0) extractor = cv::ORB::create();
    else if (descriptorType.compare("FREAK") == 0) extractor = cv::xfeatures2d::FREAK::create();
    else if (descriptorType.compare("AKAZE") == 0) extractor = cv::AKAZE::create();
    else if (descriptorType.compare("SIFT") == 0) extractor = cv::xfeatures2d::SIFT::create();
    else throw invalid_argument("Unknown descriptorType" + descriptorType);



    // perform feature description
    double t = (double) cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
//    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

```

#### MP.5 Descriptor Matching

```c++
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType,
                      std::string selectorType) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    int normType;
    if (descriptorType.compare("DES_BINARY")) normType = cv::NORM_HAMMING;
    else if (descriptorType.compare("DES_HOG")) normType = cv::NORM_L2;
    else throw invalid_argument("Unknown descriptorType " + descriptorType);

    if (matcherType.compare("MAT_BF") == 0) matcher = cv::BFMatcher::create(normType, crossCheck);
    else if (matcherType.compare("MAT_FLANN") == 0){
        if(normType==cv::NORM_HAMMING)
        {
            const cv::Ptr<cv::flann::IndexParams>& indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
        }
        else matcher = cv::FlannBasedMatcher::create();
    }
    else throw invalid_argument("Unknown matcherType " + matcherType);

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0) { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    } else if (selectorType.compare("SEL_KNN") == 0) { // k nearest neighbors (k=2)
        int k = 2;
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        // Filter knn matches by descriptor distance ratio
        double minDescDistRatio = 0.8;
        for (auto matchPair : knn_matches) {
            if (matchPair[0].distance < minDescDistRatio * matchPair[1].distance) {
                matches.push_back(matchPair[0]);
            }
        }
    }
}
```

#### MP.6 Descriptor Distance Ratio

```c++
        // Filter knn matches by descriptor distance ratio
        double minDescDistRatio = 0.8;
        for (auto matchPair : knn_matches) {
            if (matchPair[0].distance < minDescDistRatio * matchPair[1].distance) {
                matches.push_back(matchPair[0]);
            }
        }

```

#### MP.7 Performance Evaluation 1

###### Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

| Detectors | Number of Keypoints |
| ---------------|----------------|
|SHITOMASI| 1179 |
|HARRIS| 248 |
|FAST| 4094 |
|BRISK| 2762 |
|ORB| 1161 |
|AKAZE| 1670 |
|SIFT| 1386 |

Comments on keypoints neighborhood: 
* SIFT and AKAZE have similar distribution of keypoints.
* BRISK and ORB has similar distribution of keypoints.
* FAST have more keypoints than SHOTOMASI with shorter time

#### MP.8 Performance Evaluation 2
######Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

| Detectors+Descriptor | Number of Matched Keypoints |
| ---------------|----------------|
|SHITOMASI+BRISK| 767 |
|SHITOMASI+BRIEF| 944 |
|SHITOMASI+ORB| 907 |
|SHITOMASI+FREAK| 768 |
|SHITOMASI+AKAZE| N/A|
|SHITOMASI+SIFT| 927 |
|HARRIS+BRISK| 142 |
|HARRIS+BRIEF| 173 |
|HARRIS+ORB| 160 |
|HARRIS+FREAK| 144 |
|HARRIS+FREAK| N/A|
|HARRIS+SIFT| 163 |
|FAST+BRISK| 2183 |
|FAST+BRIEF| 2831 |
|FAST+ORB| 2762 |
|FAST+FREAK| 2233 |
|FAST+AKAZE| N/A|
|FAST+SIFT| 2782 |
|BRISK+BRISK| 1570 |
|BRISK+BRIEF| 1704 |
|BRISK+ORB| 1510 |
|BRISK+FREAK| 1524 |
|BRISK+AKAZE| N/A|
|BRISK+SIFT| 1646 |
|ORB+BRISK| 751 |
|ORB+BRIEF| 545 |
|ORB+ORB| 761 |
|ORB+FREAK| 420 |
|ORB+AKAZE| N/A|
|ORB+SIFT| 763 |
|AKAZE+BRISK| 1215 |
|AKAZE+BRIEF| 1266 |
|AKAZE+ORB| 1186 |
|AKAZE+FREAK| 1187 |
|AKAZE+AKAZE| 1259 |
|AKAZE+SIFT| 1270 |
|SIFT+BRISK| 592 |
|SIFT+BRIEF| 702 |
|SIFT+ORB| error |
|SIFT+FREAK| 593 |
|SIFT+AKAZE| N/A|
|SIFT+SIFT| 800 |

#### MP.9 Performance Evaluation 3
###### Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.


 | Detectors+Descriptor | Time |
 | ---------------|----------------|
 |SHITOMASI+BRISK| 168.312 ms|
 |SHITOMASI+BRIEF| 170.798 ms|
 |SHITOMASI+ORB| 183.208 ms|
 |SHITOMASI+FREAK| 139.767 ms|
 |SHITOMASI+AKAZE | N/A ms|
 |SHITOMASI+SIFT| 223.195 ms|
 |HARRIS+BRISK| 176.408 ms|
 |HARRIS+BRIEF| 189.947 ms|
 |HARRIS+ORB| 180.507 ms|
 |HARRIS+FREAK| 211.921 ms|
 |HARRIS+FREAK| N/A ms|
 |HARRIS+SIFT| 184.814 ms|
 |FAST+BRISK| 40.7903 ms| 
 |FAST+BRIEF| 44.0156 ms| 
 |FAST+ORB| 63.9577 ms| 
 |FAST+FREAK| 70.5224 ms| 
 |FAST+AKAZE| N/A|
 |FAST+SIFT| 51.1268 ms|
 |BRISK+BRISK| 2730.06 ms| 
 |BRISK+BRIEF| 2794.21 ms| 
 |BRISK+ORB| 3454.98 ms| 
 |BRISK+FREAK| 3170.98 ms| 
 |BRISK+AKAZE| N/A|
 |BRISK+SIFT| 2844.48 ms| 
 |ORB+BRISK| 99.3423 ms|
 |ORB+BRIEF| 96.4621 ms|
 |ORB+ORB| 101.593 ms|
 |ORB+FREAK| 90.2993 ms|
 |ORB+AKAZE| N/A|
 |ORB+SIFT| 102.04 ms|
 |AKAZE+BRISK| 744.039 ms| 
 |AKAZE+BRIEF| 772.091 ms| 
 |AKAZE+ORB| 850.183 ms| 
 |AKAZE+FREAK| 746.323 ms| 
 |AKAZE+AKAZE| 768.481 ms| 
 |AKAZE+SIFT| 1133.32 ms| 
 |SIFT+BRISK| 1388.96 ms|
 |SIFT+BRIEF| 1346.09 ms|
 |SIFT+ORB| error: (-215:Assertion failed) inv_scale_x > 0|
 |SIFT+FREAK| 1281.6 ms|
 |SIFT+AKAZE| N/A|
 |SIFT+SIFT| 1409.91 ms|

  My prefferance based on the data:
  
  | Top 3 Detectors+Descriptor | Reason |
| ---------------|----------------|
|FAST+BRISK| Fastest and have more matched point|
|FAST+BRIEF| Again second Fastest and have more matched point|
|FAST+SIFT| Fast and have most matched point but SIFT may require licensing so put it on third|



