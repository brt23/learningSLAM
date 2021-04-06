#include <iostream>
#include <chrono>
#include "opencv2/opencv.hpp"


int main(int argc, char** argv)
{
    cv::String imagePath_1 = "/home/y/文档/Code/learningSLAM/slambook2/ch7/1.png";
    cv::String imagePath_2 = "/home/y/文档/Code/learningSLAM/slambook2/ch7/2.png";

    // 读取图像
    cv::Mat image_1 = cv::imread(imagePath_1, cv::IMREAD_COLOR);
    cv::Mat image_2 = cv::imread(imagePath_2, cv::IMREAD_COLOR);
    assert(image_1.data != nullptr && image_2.data != nullptr);

    // 初始化
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // 第一步：检测Oriented Fast角点位置
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(image_1, keypoints_1);
    detector->detect(image_2, keypoints_2);

    // 第二步：根据焦点位置计算BRIEF描述子
    descriptor->compute(image_1, keypoints_1, descriptors_1);
    descriptor->compute(image_2, keypoints_2, descriptors_2);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost = " << time_used.count() << " seconds." << std::endl;

    cv::Mat outImage_1;
    cv::drawKeypoints(image_1, keypoints_1, outImage_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features", outImage_1);
    
    // 第三步：对两幅图像中的BREIF描述子进行匹配，使用Hamming距离
    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 -t1);
    std::cout << "match ORB cost = " << time_used.count() << " seconds." << std::endl;
    std::printf("M: %ld, D1: %d, D2: %d\n", matches.size(), descriptors_1.rows, descriptors_2.rows);

    // 第四步：匹配点筛选
    // 计算最小距离和最大距离
    auto min_max = std::minmax_element(matches.begin(), matches.end(),
        [](const cv::DMatch &m1, const cv::DMatch &m2){return m1.distance < m2.distance;});
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    std::printf("-- Max dist : %f \n", max_dist);
    std::printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍最小距离时，即认为匹配有误。但有时最小距离会非常小，所以要设置一个经验值30作为下限
    std::vector<cv::DMatch> good_matches;
    for (int i=0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    // 第五步：绘制匹配结果
    cv::Mat image_matched;
    cv::Mat image_goodmatched;
    cv::drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches, image_matched);
    cv::drawMatches(image_1, keypoints_1, image_2, keypoints_2, good_matches, image_goodmatched);
    cv::imshow("all matches", image_matched);
    cv::imshow("good matches", image_goodmatched);
    cv::waitKey(0);

    return 0;
}