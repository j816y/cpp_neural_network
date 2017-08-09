#include "ann.h"

//==================== Activation functions ======================
double Ann::sigmoid(double input){
	return 1.0/(1.0+exp(-input));
}
double Ann::step(double input){
	if (input > 0)
		return 1;
	else
		return 0;
		
}
double Ann::relu(double input){
	//Relu(x) = max(0,x)
	if (input < 0)
		return 0;
	else	//else do nothing
		return input;
}
double Ann::activation(ActType actType, double input){
	switch(actType){
		case 0:
			return sigmoid(input);
			break;
		case 1:
			return step(input);
			break;
		case 2:
			return relu(input);
			break;
	}
}

std::vector<double>* Ann::getInputPtr(){
	return inputPtr;
}
std::vector<double>* Ann::getOutputPtr(){
	return outputPtr;
}
std::vector<std::vector<std::vector<double> > >* Ann::getWeightPtr(){
	return weightPtr;
}

void Ann::readHeader(){
	// Reading file headers
    char number;
    for (int i = 0; i < 16; i++) {
        image.read(&number, sizeof(char));
	}
    for (int i = 0; i < 8; i++) {
        label.read(&number, sizeof(char));
	}
}

void Ann::initAnn(){
	srand(time(NULL));
	
	inputVector.resize(IN_SIZE);
	expectedValue.resize(OUT_SIZE);
	outputVector.resize(OUT_SIZE);
	inputVector[IN_SIZE-1] = BIAS;

	int nLayers = numOfLayers - 1;		//number of hidden layers + output layer
	//weight initialization
	(*weightPtr).resize(nLayers);
	(*deltaPtr).resize(nLayers);
	(*thetaPtr).resize(nLayers);
	(*layerOutPtr).resize(nLayers);
    for (int i = 0; i < nLayers; i++){
        //weight stored in the following format: w[layer_index][neuron_index][input_index]
        (*weightPtr)[i].resize(numOfNeurons[i+1]);
        (*deltaPtr)[i].resize(numOfNeurons[i+1]);

        for (int j = 0; j < numOfNeurons[i+1]; j++){	//number of neurons for the next layer (where the computation happens)
        	(*thetaPtr)[i].push_back(0);
        	(*layerOutPtr)[i].push_back(0);
            for (int k = 0; k < numOfNeurons[i]; k++){	//number of neurons for current layer (where the input/intermediate output generates)
				bool sign = (bool)(rand() % 2);
				(*weightPtr)[i][j].push_back((double)(rand() % 6) / 10.0);
				if(sign)
					(*weightPtr)[i][j][k] = -(*weightPtr)[i][j][k];
                (*deltaPtr)[i][j].push_back(0);
                if(DEBUG_MODE)
                	//set all weights as the neuron index
                	//for instance, every weight goes to neuron 1 is 1
                	(*weightPtr)[i][j][k] = j + 1;
            }
        }
    }
    
    if(DEBUG_MODE){
    	for (int i = 0; i < (*weightPtr).size(); i++){
    		//starts with the first hidden layer
    		std::cout << "Layer " << i+1 << ":	" << (*weightPtr)[i].size() << " neurons." << std::endl;
	        for (int j = 0; j < (*weightPtr)[i].size(); j++){
	        	std::cout << "	Neuron " << j+1 << ":	" << (*weightPtr)[i][j].size() << " weights.\n";
	        	// //print weight for each neuron
	        	// for (int k = 0; k < weight[i][j].size(); k++){
		        // 	std::cout << weight[i][j][k] << " ";
		        // 	if (k!= 0 && k%10 == 0)		//10 weights per row
		        // 		std::cout << std::endl;
		        // }
		        // std::cout << std::endl;
	        }
	    }
	    getchar();
    }
    
}

void Ann::loadInput(){
	// Read img, binarzied grey scale value.
    char number;
    int img[WIDTH][HEIGHT];
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
        	if (binInput){
        		image.read(&number, sizeof(char));
				if (number == 0) {
					img[i][j] = 0; 
				} else {
				img[i][j] = 1;
				}
        	}
			else{
				//for non-binarized
				image.read(&number, sizeof(char));
				img[i][j] = abs((int)number);	
			}
        }
	}
	//convert 2D image array to 1D inputVector
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int position = j + (i*WIDTH);
            inputVector[position] = img[i][j];
            if(FFDEBUG)
            	//for debugging, use index value for each input
            	inputVector[position] = position + 1;
        }
	}
	// Reading label
    label.read(&number, sizeof(char));
    for (int i = 0; i < OUT_SIZE; i++) {
		expectedValue[i] = 0.0;
	}
    expectedValue[number] = 1.0;

    if(DEBUG_MODE){
		// print image
		std::cout << "Image:" << std::endl;
		for (int i = 0; i < HEIGHT; i++) {
			for (int j = 0; j < WIDTH; j++) {
				if(img[i][j] == 0)
					std::cout << " ";
				else
					std::cout << "@";
			}
			std::cout << std::endl;
		}
		// Print expected value & label
	    std::cout << "Label: " << (int)(number) << std::endl;
	    for (int i = 0; i<expectedValue.size(); i++){
	    	std::cout << expectedValue[i];
	    }
	    std::cout << std::endl;
		//getchar();
	}
}

void Ann::feedForward(){
	std::vector<double> tempIn;	//an intermediate input for each layer
	std::vector<double> tempOut;	//an intermediate output for each layer
	
	tempIn.resize((*inputPtr).size());
	tempIn = (*inputPtr);
	
	int nLayers = (*weightPtr).size();		//number of hidden layers + output layer
	for (int i = 0; i < nLayers; i++) {
		int nNeurons = (*weightPtr)[i].size();	//number of neurons in each layer
		tempOut.resize(nNeurons);
		if(FFDEBUG){	//DEBUG_MODE
			std::cout << "Layer " << i+1 << std::endl;
			std::cout << "tempIn Size:	" << tempIn.size() << std::endl;
			std::cout << "tempOut Size:	" << tempOut.size() << std::endl;
			getchar();
		}
		if(FFDEBUG){
			debugReport << "Layer " << i+1 << std::endl;	
		}
		for (int j = 0; j < nNeurons; j++) {
			int nInputs = (*weightPtr)[i][j].size();//number of inputs connected to one neuron
			tempOut[j] = 0;							//initialize tempOut
			for (int k = 0; k < nInputs; k++) {
				tempOut[j] += tempIn[k] * (*weightPtr)[i][j][k];
				
				if(FFDEBUG){	//DEBUG_MODE
					std::cout << "Neuron " << j+1 << ", Input " << k+1 << ": \n";
					std::cout << "Input:		" << tempIn[k] << std::endl;
					std::cout << "Weight:		" << (*weightPtr)[i][j][k] << std::endl;
					std::cout << "Accumulation:	" << tempOut[j] << std::endl;
					getchar();
				}
				if(FFDEBUG){
					debugReport << "Input " << k << ":	" << tempIn[k] << "*" << (*weightPtr)[i][j][k] << "=" << tempIn[k] * (*weightPtr)[i][j][k] <<std::endl;
					//output (before activation)
					debugReport << "Out(b) " << j+1 << ":	" << tempOut[j] << std::endl;
				}
			}
			tempOut[j] = activation(actType, tempOut[j]);
			if(FFDEBUG){
				//output (after activation)
				debugReport << "Out(a) " << j+1 << ":	" << tempOut[j] << std::endl;
			}
		}
		tempIn.resize(tempOut.size());
		tempIn = tempOut;	//the intermediate output becomes an input for next layer
		layerOut[i] = tempOut;
	}
	(*outputPtr) = tempOut;
}

void Ann::backPropagation(){
	//----------------- Hard coded version for debugging -----------------------
	//note: instead of i j k, use a more meaningful temp_variable since it gets very complicated.
	double sum;
	for (int i = 0; i < OUT_SIZE; i++){
		(*thetaPtr)[1][i] = layerOut[1][i] * (1-layerOut[1][i]) * (expectedValue[i]-layerOut[1][i]);
	}
	for (int i = 0; i < OUT_SIZE; i++){
		for (int j = 0; j < 128; j++){
			//layerOut[0][j] is the output of hidden layer, size should be 128
			(*deltaPtr)[1][i][j] = (learningRate * (*thetaPtr)[1][i] * layerOut[0][j]) + (momentum * (*deltaPtr)[1][i][j]);
			(*weightPtr)[1][i][j] += (*deltaPtr)[1][i][j];
		}
	}

	//the reverse of j and i needs some graphic explanation
	//i is the one needed to be repeating
	for (int j = 0; j < 128; j++){
		sum = 0.0;
		for (int i = 0; i < OUT_SIZE; i++){
			sum += (*weightPtr)[1][i][j] * (*thetaPtr)[1][i];
		}
		(*thetaPtr)[0][j] = layerOut[0][j] * (1-layerOut[0][j]) * sum;
	}

	for (int i = 0; i < 128; i++){
		for (int j = 0; j < IN_SIZE; j++){
			(*deltaPtr)[0][i][j] = (learningRate * (*thetaPtr)[0][i] * (*inputPtr)[j]) + (momentum * (*deltaPtr)[0][i][j]);
			(*weightPtr)[0][i][j] += (*deltaPtr)[0][i][j];
		}
	}
	
	
	//------------------- A generic version that doesn't learn -----------------
	
	
	// double sum;
	// int nLayers = (*weightPtr).size()-1;	//number of sets of weight matrix between layers
	//calculate theta from output to the last hidden layer
	for (int oSize = 0; oSize < OUT_SIZE; oSize++)
		(*thetaPtr).back()[oSize] = (*layerOutPtr).back()[oSize] * (1 - (*layerOutPtr).back()[oSize]) * (expectedValue[oSize] - (*layerOutPtr).back()[oSize]);	
	
	for (int nLayers = (numOfNeurons.size()-1); nLayers >= 0; nLayers--){	//number of layers, indexs starts from the output layer
		if(BPDEBUG)
			std::cout << "Layer:	" << nLayers << std::endl;
		for (int nNeurons = 0; nNeurons < numOfNeurons[nLayers]; nNeurons++){	//number of neurons in each layer
			if(BPDEBUG)
				std::cout << "Neurons:	" << nNeurons << std::endl;
		}
	}
	
	// for (int i = 0; i < (*weightPtr)[outInd].size(); i++){
	// 	sum = 0;
	// 	for (int j = 0; j < (*weightPtr)[outInd][i].size(); j++){
	// 		sum += (*weightPtr)[outInd][i][j] * (*thetaPtr)[outInd][j];
	// 	}
	// 	(*thetaPtr)[outInd-1][i] = (*layerOutPtr)[outInd-1][i] * (1-(*layerOutPtr)[outInd-1][i]) * sum;
	// }
	
	// //back propagation from hidden layers to input
	// for (int i = nLayers-1; i >= 0; i--){
	// 	int nNeurons = (*weightPtr)[i].size();	//number of neurons in each layer
	// 	for (int j = 0; j < nNeurons; j++){
	// 		int nInputs = (*weightPtr)[i][j].size();
	// 		for (int k = 0; k < (nInputs); k++){
	// 			if(i == 0)
	// 				(*deltaPtr)[i][j][k] = learningRate * (*thetaPtr)[i][j] * (*inputPtr)[j] + (momentum * (*deltaPtr)[i][j][k]);
	// 			else
	// 				(*deltaPtr)[i][j][k] = learningRate * (*thetaPtr)[i][j] * (*layerOutPtr)[i][j] + (momentum * (*deltaPtr)[i][j][k]);
	// 			(*weightPtr)[i][j][k] += (*deltaPtr)[i][j][k];
	// 		}
	// 	}
	// }
}

int Ann::learning(){
	int nIterations = 0;
	//initialize delta value
	for (int i = 0; i < delta.size(); i++){
		for (int j = 0; j < delta[i].size(); j++){
			for (int k = 0; k <delta[i][j].size(); k++){
				(*deltaPtr)[i][j][k] = 0;
			}
		}
	}
	//train until it reaches maximum epochs or error is lower than the threshold
	while(nIterations < epochs){
	//while(squareError() > epsilon){
		nIterations++;			
		feedForward();
		backPropagation();
		
		if(squareError() < epsilon)
			break;

		if(BPDEBUG)
			std::cout << "Iteration " << nIterations << ", error:	" << squareError() << std::endl;
    }
	return nIterations;
}

double Ann::squareError(){
	double rms = 0.0;
	//Root Mean Square (RMS)
    for (int i = 0; i < OUT_SIZE; i++) {
        rms += ((*outputPtr)[i] - expectedValue[i])*((*outputPtr)[i] - expectedValue[i]);
	}
    rms *= 0.5;
    return rms;
}

void Ann::trainReport(int sample, int nIterations, double error){
	report << "Sample " << sample+1 << ":	No. iterations:	" << nIterations << ".	Error = " << error << std::endl;
}

void Ann::writeWeight(std::string file_name) {
    std::ofstream file(file_name.c_str(), std::ios::out);
	
	for (int i = 0; i < (*weightPtr).size(); i++){
		for (int j = 0; j < (*weightPtr)[i].size(); j++){
			for (int k = 0; k < (*weightPtr)[i][j].size(); k++){
				file << (*weightPtr)[i][j][k] << " ";
			}
			file << std::endl;
		}
	}
	file.close();
}

void Ann::summary() {
	// Details
	std::cout << "**************************************************" << std::endl;
	std::cout << "*********** Training Neural Network **************" << std::endl;
	std::cout << "**************************************************" << std::endl;
	std::cout << std::endl;
	std::cout << "Debug Mode:		";
	if(DEBUG_MODE)
		std::cout << "On" << std::endl;
	else
		std::cout << "Off" << std::endl;
	std::cout << "Binizared input:	";
	if(binInput)
		std::cout << "True" << std::endl;
	else
		std::cout << "False" << std::endl << std::endl;
	std::cout << "No. input neurons:	" << IN_SIZE << std::endl;
	for (int i = 1; i < numOfLayers-1; i++)	//1st layer is input and last layer is output
		std::cout << "No. hidden neurons:	" << numOfNeurons[i] << std::endl;
	std::cout << "No. output neurons:	" << OUT_SIZE << std::endl;
	std::cout << std::endl;
	std::cout << "No. iterations:		" << epochs << std::endl;
	std::cout << "Learning rate:		" << learningRate << std::endl;
	std::cout << "Momentum:		" << momentum << std::endl;
	std::cout << "Epsilon:		" << epsilon << std::endl;
	std::cout << "**************************************************" << std::endl;
	std::cout << "Press any key to continue.\n" << std::endl;
    getchar();
}

void Ann::loadWeight(std::string data){
	std::ifstream file(data.c_str(), std::ios::in);
	for (int i = 0; i < (*weightPtr).size(); i++){
		for (int j = 0; j < (*weightPtr)[i].size(); j++){
			for (int k = 0; k < (*weightPtr)[i][j].size(); k++){
				file >> (*weightPtr)[i][j][k];
			}
		}
	}
	file.close();
}

bool Ann::prediction(int sampleInd){
	int predictInd = 0;
	int label = 0;
	bool result;
	if(TDEBUG)
		std::cout << "Output: ";
	for (int i = 0; i < (*outputPtr).size(); i++){
		if(TDEBUG)
			//print output values
			std::cout << "Value " << i << ": " << (*outputPtr)[i] << std::endl;
			
		if((*outputPtr)[i] > (*outputPtr)[predictInd]){
			predictInd = i;
			//look for the max value from output vector
			//and store the max value index
		}
	}

	for (int i = 0; i < expectedValue.size(); i++){
		if(expectedValue[i] == 1){
			label = i;
		}
	}
	
	double error = squareError();
	printf("Error: %0.6lf\n", error);
	std::cout << "Label: " << label << ". Predicted Value: " << predictInd << ".\n";
	if (label == predictInd) {
		result = true;
		std::cout << "Correct" << std::endl;
	}
	else{
		result = false;
		std::cout << "Inorrect" << std::endl;
		if(TDEBUG){
			//print image
			std::cout << "Test Image:" << std::endl;
			for (int j = 0; j < HEIGHT; j++) {
				for (int i = 0; i < WIDTH; i++) {
					if((*inputPtr)[i+j*WIDTH] == 0)
						std::cout << " ";
					else
						std::cout << "#";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			getchar();	
		}
	}
	report << "Sample " << sampleInd << ":	";
	if(result)
		report << "Correct.	";
	else
		report << "Incorrect.	";
	report << "Label:	" << label << ". Predict:	" << predictInd << ". Error:	" << error << std::endl;
	return result;
}

void Ann::testReport(int nCorrect, double accuracy){
	report << "========= Summary ===========" << std::endl;
    report << "Correctness:	" << nCorrect << " / " << testSize << std::endl;
    report << "Accuracy:		" << accuracy << std::endl;
}