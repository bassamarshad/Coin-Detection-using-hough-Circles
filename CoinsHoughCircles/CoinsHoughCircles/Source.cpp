#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include<iostream>
using namespace cv;
using namespace std;

float getCircularityThresh(vector<Point> cntr);


int main()
{
	Mat img, gray;
	img = imread("3.jpg");

	cvtColor(img, gray, CV_BGR2GRAY);



	// smooth it, otherwise a lot of false circles may be detected
	GaussianBlur(gray, gray, Size(3, 3), 1.5, 1.5);

	namedWindow("Blurred", 0);
	imshow("Blurred", gray);

	//Calculate the Otsu threshold values
	Mat opimg = Mat(gray.rows, gray.cols, CV_8UC1);
	double otsu_thresh_val = threshold(
		gray, opimg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU
		);

	imshow("otsu", opimg);

	double high_thresh_val = otsu_thresh_val,
		lower_thresh_val = otsu_thresh_val * 0.5;

	Mat edges;

	Canny(gray, edges, lower_thresh_val, high_thresh_val, 3);



	//Morphology
	//Dilation
	int dilation_type = MORPH_RECT, dilation_size = 1;  // dilation_type = MORPH_RECT,MORPH_CROSS,MORPH_ELLIPSE
	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));

	morphologyEx(edges, edges, MORPH_DILATE, element);

	//Canny to thin the edges after dilation
	Canny(edges, edges, 50, 200, 3);

	namedWindow("Canny", 0);
	imshow("Canny", edges);


	// Double check for the circles - Just the edge image at this moment produces a lot of false circles - when the Hough circles function is run
	// Shortlisting good circle candidates by doing a contour analysis
	vector<vector<Point>> contours, contoursfil;
	vector<Vec4i> hierarchy;
	Mat contourImg1 = Mat::ones(edges.rows, edges.cols, edges.type());
	Mat contourImg2 = Mat::ones(edges.rows, edges.cols, edges.type());

	//Find all contours in the edges image
	findContours(edges.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	float circThresh;

	for (int j = 0; j < contours.size(); j++)
	{
		//Only give me contours that are closed (i.e. they have 0 or more children : hierarchy[j][2] >= 0) AND contours that dont have any parent (i.e. hierarchy[j][3] < 0 )
		if ((hierarchy[j][2] >= 0) && (hierarchy[j][3] < 0))
		{
			circThresh = getCircularityThresh(contours[j]);
			// Doing a quick compactness/circularity test on the contours P^2/A for the circle the perfect is 12.56 .. we give some room as we mostly are extracting elliptical shapes also
			if ((circThresh > 10) && (circThresh <= 30))
			{
				contoursfil.push_back(contours[j]);
			}
		}
	}


	//	drawContours(contourImg1, contours, 0, CV_RGB(255,255,255), 1, 8,0);
	//	imshow("Contour Image", contourImg1);

	for (int j = 0; j < contoursfil.size(); j++)
	{
		drawContours(contourImg2, contoursfil, j, CV_RGB(255, 255, 255), 1, 8);
	}
	namedWindow("Contour Image Filtered", 0);
	imshow("Contour Image Filtered", contourImg2);



	// good values for param-2 - for a image having circle contours = (75-90) ... for the edge image - (100-120)
	vector<Vec3f> circles;
	//HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 2, gray.rows / 8, 200, 120);
	HoughCircles(contourImg2, circles, CV_HOUGH_GRADIENT, 2, gray.rows / 8, 200, 30,15,45);

	//struct to sort the vector of pairs <int,double> based on the second double value
	struct sort_pred {
		bool operator()(const Vec3f &left, const Vec3f &right) {
			return left[2]< right[2];
		}
	};
	//sort in descending
	std::sort(circles.rbegin(), circles.rend(), sort_pred());

	float largestRadius = circles[0][2];
	float change = 0;
	float ratio;

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		float radius = circles[i][2];
		// draw the circle center
		circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		circle(img, center, radius, Scalar(0, 0, 255), 2, 8, 0);
		rectangle(img, Point(center.x - radius - 5, center.y - radius - 5), Point(center.x + radius + 5, center.y + radius + 5), CV_RGB(0, 0, 255), 1, 8, 0); //Opened contour - draw a green rectangle arpund circle
		ratio = ((radius*radius) / (largestRadius*largestRadius));
		//cout << ratio << "\n";

		//Using an area ratio based discrimination .. after some trial and error with the diff sizes ... this gives good results.
		if (ratio >= 0.85)
		{
			putText(img, "Quarter", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .25;
		}
		else if ((ratio >= 0.68) && (ratio<85))
		{
			putText(img, "5-cents", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .05;
		}
		else if ((ratio >= 0.60) && (ratio<.68))
		{
			putText(img, "1-cent", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .01;
		}
		else if ((ratio >= 0.40) && (ratio<.60))
		{
			putText(img, "Dime", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .1;
		}

	}

	putText(img, "Total Money:" + to_string(change), Point(img.cols / 10, img.rows - img.rows / 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.6, CV_AA);

	namedWindow("circles", 0);
	imshow("circles", img);

	waitKey();

	return 0;
}

float getCircularityThresh(vector<Point> cntr)
{
	float perm, area;

	perm = arcLength(Mat(cntr), true);
	area = contourArea(Mat(cntr));

	return ((perm*perm) / area);

}