#include "ann.h"

int main(){
    //======================Set Network Size=============================
    std::string defaultSetting;
    int userInput;
    
    //default is a neural network with single hidden layer and 128 neurons
    bool training = true;
    int numOfLayers = 3;
    std::vector<int> numOfNeurons;
    numOfNeurons.resize(numOfLayers);
    numOfNeurons[0] = IN_SIZE;
    numOfNeurons[1] = HL_SIZE;
    numOfNeurons[2] = OUT_SIZE;
    
    std::cout << "**************************************************" << std::endl;
    std::cout << "**************** Default Setting *****************" << std::endl;
    std::cout << "**************************************************" << std::endl;
    std::cout<< "Number of Layers: " << numOfLayers << std::endl;
    for (int i = 0; i < numOfLayers; i++){
        std::cout<< "Number of Neurons in Layer " << i+1 << ": " << numOfNeurons[i] << std::endl;
    }
    std::cout<< "Use the default setting? (Y/N):";
    std::cin >> defaultSetting;
    
    if (defaultSetting == "Y" || defaultSetting == "y"){
        std::cout << "Default setting chose. Press any key to continue.\n" << std::endl;
        getchar();
    }
    else{
        std::cout<< "Please enter number of layers (input layer included): ";
        std::cin >> numOfLayers;
        for (int i = 1; i <= numOfLayers; i++){
            std::cout<< "Please enter number of neurons for layer " << i << ": ";
            std::cin >> userInput;
            numOfNeurons.push_back(userInput);
        }
    }
    
    //============Create Network====================
	Ann training_nn(sigmoidFunct, numOfLayers, numOfNeurons, training);
    //Ann training_nn (reluFunct, numOfLayers, numOfNeurons, training);
	training_nn.summary();        //print basic network information
	training_nn.readHeader();   //get rid of header from image file (for MNIST)
	training_nn.initAnn();      
	
	//================Traing========================
	//go through the entire dataset, for MNIST is 60000
    for (int sample = 0; sample < trainSize; sample++) {
        std::cout << "Sample " << sample+1 << ": ";
        int nIterations = 0;
        double error = 0;
        
        training_nn.loadInput();
        nIterations = training_nn.learning();
        // Write down the squared error
        std::cout << "No. iterations: " << nIterations << ". ";
        error = training_nn.squareError();
        printf("Error: %0.6lf\n", error);
        training_nn.trainReport(sample, nIterations, error);
        
        // Save the current weight matrix every 100 samples
        // In case of termination
        if (sample != 0 && sample % 100 == 0) {
            std::cout << "Saving weight to " << weight_data << ".\n";
            training_nn.writeWeight(weight_data);
        }
    }
    //save final network
    training_nn.writeWeight(weight_data);
    std::cout << "Training Completed.\n";
    return 0;
}