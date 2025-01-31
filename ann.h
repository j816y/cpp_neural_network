#ifndef ANN_H
#define ANN_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <ostream>
#include <math.h>
#include <vector>
#include <string>

//for MNIST
#define BIAS		1
#define WIDTH		28					//Input Image Size
#define HEIGHT		28
#define IN_SIZE		WIDTH*HEIGHT+BIAS	//Input Size Default
#define OUT_SIZE	10					//Output Size Default
#define	HL_SIZE		64					//Hidden Layer Size Default
#define DEBUG_MODE	false				//general debug mode for init and load input
#define FFDEBUG		false				//for feedfwd debugging report generation
#define BPDEBUG		false				//for backPropagationk debugging
#define	TDEBUG		false				//for neural network test debugging

// Weights file name
const std::string weight_data = "weight_data.dat";
// Report file name
const std::string train_report = "training-report.dat";
const std::string test_report = "testing-report.dat";
const std::string debug_report = "debug-report.dat";

std::vector<int> readConfig(std::string fileName);

enum ActType{
	sigmoidFunct,	//0
	stepFunct,		//1
	reluFunct		//2
};

enum DataSet{
	MNIST,
	EMNIST,
	CIFAR10,
	CIFAR100
};

class Ann{
	std::ifstream image;
	std::ifstream label;
	std::ofstream report;
	std::ofstream debugReport;
	
	ActType actType;	//activation function types
	bool training;		//true as training, false as testing
	bool binInput;		//binaized input, default as true
	
	int numOfLayers;	//total number of layers, include input layer
	//number of neuron for each layer, vector size based on number of layers
	std::vector<int> numOfNeurons;
	
	// Training image file name
	std::string train_image;
	// Training label file name
	std::string train_label;
	// Number of training samples
	int trainSize;
	// Testing image file name
	std::string test_image;
	// Testing label file name
	std::string test_label;
	// Number of testing samples
	int testSize;
	
	//set default value in constructor
	int epochs;
	double learningRate;
	double momentum;
	double epsilon;
	
	//1D vector stored the expected output for each output neuron
	//Value is either 1 or 0, only one value is 1 and else are 0s.
	//e.g. for image "5", the expected output is 0000010000
	//where the MSB represents the probability of image "0"
	//and the LSB represents the probability of image "9"
	//used for training and accuracy calculation
	std::vector<double> expectedValue;
	
	std::vector<double> inputVector;
	std::vector<double> *inputPtr;
	std::vector<double> outputVector;
	std::vector<double> *outputPtr;
	
	//stores outputs for each layer for back propagation
	std::vector<std::vector<double> > layerOut;
	std::vector<std::vector<double> >* layerOutPtr;
	
	//3D vector to store all weight in neural network
	// vector[layer_index][neuron_index][weight_index]
	std::vector<std::vector<std::vector<double> > > weight;
	std::vector<std::vector<std::vector<double> > >* weightPtr;
	
	//different between new weight and old weight for all weight, delta should have the same size as weight
	std::vector<std::vector<std::vector<double> > > delta;
	std::vector<std::vector<std::vector<double> > >* deltaPtr;
	//different between new output and old output for each layer, theta[0] should have the same size as output from the first hidden layer, and theta[last] should have the same size at OutputVector
	std::vector<std::vector<double> > theta;
	std::vector<std::vector<double> >* thetaPtr;
	
	public:
		inline Ann(ActType, int, std::vector<int>, bool);
		inline ~Ann();
		
		//types of activation function
		double step(double input);
		double sigmoid(double input);
		double relu(double input);
		
		//choose an activation function
		double activation(ActType actType, double input);
		
		int getTrainSize();
		int getTestSize();
		
		std::vector<double>* getInputPtr();
		std::vector<double>* getOutputPtr();
		std::vector<std::vector<std::vector<double> > >* getWeightPtr();
		
		void chooseDataset(DataSet data);
		void readHeader();						//get rid of header from image file (for MNIST)
		void initAnn();							//set vectors size, assign randomized weight with range(-1,1)
		void loadInput();						//read input (print is disabled by default)
		void feedForward();
		void backPropagation();						
		int learning();							//calls feedfwd & backpropa for n times
		double squareError();					//return error
		void trainReport(int sample, int nIterations, double error);
		void writeWeight();						//update weight for each iteration
		void summary();							//print basic network information
		
		//Function for Testing
		void loadWeight();		//load trained weight for testing
		bool prediction(int sampleInd);			//
		void testReport(int nCorrect, double accuracy);	//write report
};

//constructor
Ann::Ann(ActType act, int layers, std::vector<int> neurons, bool train){
	actType = act;
	training = train;
	inputPtr = &inputVector;
	outputPtr = &outputVector;
	weightPtr = &weight;
	deltaPtr = &delta;
	thetaPtr = &theta;
	layerOutPtr = &layerOut;
	numOfLayers = layers;
	numOfNeurons = neurons;
	
	//default value
	binInput = true;
	epochs = 512;
	learningRate = 0.01;
	momentum = 0.9;
	epsilon = 1e-4;
	
	debugReport.open(debug_report.c_str(), std::ios::out);
}

//destructor
Ann::~Ann(){
	// if(inputPtr != NULL){
	// 	free(inputPtr);	
	// }
	// if(outputPtr != NULL){	
	// 	free(outputPtr);
	// }
	debugReport.close();
	report.close();
    image.close();
    label.close();
}
#endif