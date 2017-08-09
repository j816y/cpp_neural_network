#include "ann.h"

int main(){
    //default network size, should be stored in a training file
    bool training = false;
    int numOfLayers = 3;
    std::vector<int> numOfNeurons;
    numOfNeurons.resize(numOfLayers);
    numOfNeurons[0] = IN_SIZE;
    numOfNeurons[1] = HL_SIZE;
    numOfNeurons[2] = OUT_SIZE;
    
    Ann testing_nn (sigmoidFunct, numOfLayers, numOfNeurons, training);
    testing_nn.summary();           //print basic network information
	testing_nn.readHeader();        //get rid of header from image file (for MNIST)
	testing_nn.initAnn();
    testing_nn.loadWeight(weight_data);
	
	int nCorrect = 0;		//number of correct prediction
	bool result;
    for (int sample = 0; sample < testSize; sample++) {
        std::cout << "Sample " << sample+1 << std::endl;
	    testing_nn.loadInput();
	    testing_nn.feedForward();
	    result = testing_nn.prediction(sample);
	    if(result)
	        nCorrect++;
    }
    //Summary
    float accuracy = (float)(nCorrect) / (float)testSize * 100.0;
    std::cout << "========= Summary ===========" << std::endl;
    std::cout << "Correctness:	" << nCorrect << "/" << testSize << std::endl;
    std::cout << "Accuracy:	" << accuracy << std::endl;
    testing_nn.testReport(nCorrect, accuracy);
    std::cout << "========= Summary ===========" << std::endl;
    return 0;
}