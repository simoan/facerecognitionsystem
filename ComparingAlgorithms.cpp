#define radiusLBP 1       // 1
#define neighborsLBP 8   // 16
#define gridX 8
#define gridY 8
#define thresholdLBP 10000

#define sizeGauss 9  // 9 is smallest kernel
#define sigmaGauss 20  // small deviation: means not much blurring, big deviation: a lot of blurring

#define numberOfLabels 10  // uses either whole database or only family

#define expIncrement 1.3 // how much each step is more (exponential)
#define linIncrement 0
#define numberIterations 100  // how often it will be incremented

#define trainingSet 8 // t1, t2,...
#define sampleSet  4   // s1, s2,...
#define newModelTrue 1
#define liveModeTrue 0
#define imwriteTrue 0

#define sizeWidth 92   // 92, 368
#define sizeHeight 112    // 112, 448

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/face.hpp>

using namespace cv;
using namespace cv::face;
using namespace std;


// for detecting faces and pre-processing them accordingly
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;

// 
String save_results = "save.txt";

// detect face, crop, resize, gray conversion, equalization, (normalization??), gauss
void preprocessing(Mat& face)
{
	cv::cvtColor(face, face, COLOR_BGR2GRAY);
	cv::GaussianBlur(face, face, Size(sizeGauss, sizeGauss), sigmaGauss, sigmaGauss);
	equalizeHist(face, face);
}

Mat cropping(Mat face, Rect region, int circleTrue = 0)
{
	Mat face2;
	Point center(region.x + region.width / 2, region.y + region.height / 2);
	if (circleTrue)
	{
		cv::cvtColor(face, face, COLOR_BGR2GRAY);
		Mat backup(face.rows, face.cols, CV_8UC1, Scalar(0, 0, 0));
		ellipse(face, center, Size(region.width / 2, region.height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);  // face = face(ellipse)
		bitwise_and(face, backup, face2);
		imshow("AND", face);
		imwrite("C:/Users/Simon/Desktop/OpenCV/cleanup/im.jpg", face2);
	}
	else
	{
		face = face(region);
	}
	//imwrite("C:/Users/Simon/Desktop/OpenCV/cleanup/im.jpg", face);
	return face;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}
	string line, path, classlabel;
	cv::Mat loadedIm;
	std::vector<Rect> faces;
	face_cascade.load(face_cascade_name);  // load cascade to detect faces
	while (getline(file, line)) 
	{
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) 
		{
			loadedIm = imread(path, CV_LOAD_IMAGE_COLOR); 
			if ((loadedIm.rows != sizeHeight) && (loadedIm.cols != sizeWidth))   // AT&T format, they are grayscale and cropped already
			{
				face_cascade.detectMultiScale(loadedIm, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(130, 10));
				if (faces.size() == 1)   // usually this should be one face, otherwise wrong data
				{
					loadedIm = cropping(loadedIm, faces[0], 0);
					cv::resize(loadedIm, loadedIm, Size(sizeWidth, sizeHeight), 0, 0, CV_INTER_LINEAR);  // 1,1, NN better?
					if (imwriteTrue) imwrite(path, loadedIm);
					preprocessing(loadedIm);
					images.push_back(loadedIm);						 
					labels.push_back(atoi(classlabel.c_str()));  
				}
			}
			else
			{
				preprocessing(loadedIm);
				if (imwriteTrue) cv::imwrite(path, loadedIm);
				images.push_back(loadedIm);						 // images are loaded from path with imread() in grayscale and allocated next to each other
				labels.push_back(atoi(classlabel.c_str()));  // labels are for images are allocated next to each other
			}
		}
	}
}

int liveDetection(Ptr<LBPHFaceRecognizer> modelLBP, Ptr<BasicFaceRecognizer> modelFisher, Ptr<BasicFaceRecognizer> modelEigen)
{
	VideoCapture capture;
	Mat frame, crop;
	int predLabel = -1;
	double predConfidence = -1;
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	capture.open(0);
	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		//-- 3. Apply the classifier to the frame
		std::vector<Rect> faces;
		Mat frame_gray;

		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		//-- Detect faces
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t i = 0; i < faces.size(); i++)
		{
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

			crop = frame_gray(faces[i]);
			cv::GaussianBlur(crop, crop, Size(sizeGauss, sizeGauss), sigmaGauss, sigmaGauss);
			
			cv::resize(crop, crop, Size(sizeWidth, sizeHeight), 0, 0, CV_INTER_LINEAR);
			modelLBP->predict(crop, predLabel, predConfidence);
			String textLBP = "LBP" + std::to_string(predLabel) + "|" + std::to_string(predConfidence);
			modelFisher->predict(crop, predLabel, predConfidence);
			String textFisher =  "Fisher" + std::to_string(predLabel) + "|" + std::to_string(predConfidence);
			modelFisher->predict(crop, predLabel, predConfidence);
			String textEigen = "Eigen" + std::to_string(predLabel) + "|" + std::to_string(predConfidence);

			//String text = "SIm";
			center.x = center.x + 80;
			putText(frame, textLBP, center, FONT_HERSHEY_SIMPLEX, 1, 0, 2);
			center.y = center.y + 50;
			putText(frame, textFisher, center, FONT_HERSHEY_SIMPLEX, 1, 0, 2);
			center.y = center.y + 50;
			putText(frame, textEigen, center, FONT_HERSHEY_SIMPLEX, 1, 0, 2);
		}		
		//-- Show what you got
		imshow("detectedFace", frame);
		int c = waitKey(33);
		if ((char)c == 27) { break; } // escape
	}
}


int main(int argc, const char *argv[])
{
	string path_training = "C:/Users/Simon/Desktop/OpenCV/cleanup/path_t";
	path_training = path_training + std::to_string(trainingSet) + ".csv";

	string path_sample = "C:/Users/Simon/Desktop/OpenCV/cleanup/path_s";
	path_sample = path_sample + std::to_string(sampleSet) + ".csv";

	vector<Mat> images_t;    // here training database is loaded
	vector<int> labels_t;    // here corresponding labels for picture classes are loaded
	vector<Mat> images_s;    // here training database is loaded
	vector<int> labels_s;    // here corresponding labels for picture classes are loaded

	// Read in the TRAINING data
	try {
		read_csv(path_training, images_t, labels_t);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << path_training << "\". Reason: " << e.msg << endl;
		system("pause");
		exit(1);
	}

	if (images_t.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(Error::StsError, error_message);
	}

	Ptr<LBPHFaceRecognizer> modelLBP = createLBPHFaceRecognizer(radiusLBP,neighborsLBP, gridX, gridX, thresholdLBP);
	Ptr<BasicFaceRecognizer> modelFisher = createFisherFaceRecognizer();
	Ptr<BasicFaceRecognizer> modelEigen = createEigenFaceRecognizer();

	if (newModelTrue)  // create new model
	{
		modelLBP->train(images_t, labels_t);
		String saveLBP = "C:/Users/Simon/Desktop/OpenCV/cleanup/Model/modelLBP_t";
		saveLBP = saveLBP + std::to_string(trainingSet) + ".xml";
		modelLBP->save(saveLBP);

		modelFisher->train(images_t, labels_t);
		String saveFisher = "C:/Users/Simon/Desktop/OpenCV/cleanup/Model/modelFisher_t";
		saveFisher = saveFisher + std::to_string(trainingSet) + ".xml";
		modelFisher->save(saveFisher);

		modelEigen->train(images_t, labels_t);  // takes 25s, uses images from images and corresp labels
		String saveEigen = "C:/Users/Simon/Desktop/OpenCV/cleanup/Model/modelEigen_t";
		saveEigen = saveEigen + std::to_string(trainingSet) + ".xml";
		modelEigen->save(saveEigen);
	}
	else       // use existing model
	{
		String loadLBP, loadFisher, loadEigen;
		loadLBP = "C:/Users/Simon/Desktop/OpenCV/cleanup/Model/modelLBP_t";
		loadLBP = loadLBP + std::to_string(trainingSet) + ".xml";
		modelLBP->load(loadLBP);

		loadFisher = "C:/Users/Simon/Desktop/OpenCV/cleanup/Model/modelFisher_t";
		loadFisher = loadFisher + std::to_string(trainingSet) + ".xml";
		modelFisher->load(loadFisher);

		loadEigen = "C:/Users/Simon/Desktop/OpenCV/cleanup/Model/modelEigen_t";
		loadEigen = loadEigen + std::to_string(trainingSet) + ".xml";
		modelEigen->load(loadEigen);
	}

	// Read in the SAMPLE data
	try {
		read_csv(path_sample, images_s, labels_s);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << path_sample << "\". Reason: " << e.msg << endl;
		system("pause");
		exit(1);
	}

	int predLabel;
	double predConfidence;
	if (!liveModeTrue)
	{
		cout << "START!! Gauss: " << sigmaGauss << " Gauss size: " << sizeGauss <<" training: " << trainingSet << "samples: " << sampleSet << endl;
		cout << "RadiusLBP: " << radiusLBP << " surrounding neigbhors: " << neighborsLBP << " gridx: " << gridX << " gridy: " << gridY << endl;
		cout << "Local Binary Pattern" << endl;
		for (int i = 0; i < images_s.size(); i++)
		{
			modelLBP->predict(images_s[i], predLabel, predConfidence);
			String result_message = format("Predicted class = %d / Actual class = %d with %f, i %d", predLabel, labels_s[i], predConfidence, i);
			cout << result_message << endl;
		}
		cout << endl << endl;
		cout << "Eigenfaces" << endl;
		for (int i = 0; i < images_s.size(); i++)
		{
			modelEigen->predict(images_s[i], predLabel, predConfidence);
			String result_message = format("Predicted class = %d / Actual class = %d with %f, i %d", predLabel, labels_s[i], predConfidence, i);
			cout << result_message << endl;
		}
		cout << endl << endl;
		cout << "Fisher Faces" << endl;
		for (int i = 0; i < images_s.size(); i++)
		{
			modelFisher->predict(images_s[i], predLabel, predConfidence);
			String result_message = format("Predicted class = %d / Actual class = %d with %f, i %d", predLabel, labels_s[i], predConfidence, i);
			cout << result_message << endl;
		}
	}
	else
	{
		liveDetection(modelLBP, modelEigen, modelFisher);
	}

	system("pause");
}