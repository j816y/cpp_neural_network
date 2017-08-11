#include "ann.h"

int main(){
    DataSet dataset = MNIST;
    std::string configFile = "config.dat";
    
    bool training = false;
    std::vector<int> configValue;
    int numOfLayers;
    std::vector<int> numOfNeurons;
    configValue = readConfig(configFile);
    
    for (int i = 0; i < configValue.size(); i++){
        if (i == 0)
            numOfLayers = configValue[i];
        else
            numOfNeurons.push_back(configValue[i]);
    }
    
    Ann testing_nn (sigmoidFunct, numOfLayers, numOfNeurons, training);
    testing_nn.chooseDataset(dataset);
    testing_nn.initAnn();
    testing_nn.summary();           //print basic network information
	testing_nn.readHeader();        //get rid of header from image file (for MNIST)
    testing_nn.loadWeight();
	
	int nCorrect = 0;		//number of correct prediction
	bool result;
	int SampleSize = testing_nn.getTestSize();
	std::cout << "Testing:  " << SampleSize << std::endl;
    for (int sample = 0; sample < SampleSize; sample++) {
        std::cout << "Sample " << sample+1 << std::endl;
	    testing_nn.loadInput();
	    testing_nn.feedForward();
	    result = testing_nn.prediction(sample);
	    if(result)
	        nCorrect++;
    }
    //Summary
    float accuracy = (float)(nCorrect) / (float)SampleSize * 100.0;
    std::cout << "========= Summary ===========" << std::endl;
    std::cout << "Correctness:	" << nCorrect << "/" << SampleSize << std::endl;
    std::cout << "Accuracy:	" << accuracy << std::endl;
    testing_nn.testReport(nCorrect, accuracy);
    std::cout << "========= Summary ===========" << std::endl;
    return 0;
}