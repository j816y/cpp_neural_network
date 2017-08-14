#include "ann.h"

bool training = true;
//======================Set Network Size=============================
std::ofstream config;       //store network size for testing purpose
std::string config_file = "config.dat";
DataSet dataset = MNIST;
int numOfLayers;
std::vector<int> numOfNeurons;

//User Interface for network setting
void networkConstruction(){
    std::string defaultSetting;
    int userInput;
    
    std::cout << "**************************************************" << std::endl;
    std::cout << "**************** Default Setting *****************" << std::endl;
    std::cout << "**************************************************" << std::endl;
    std::cout<< "Number of Layers: " << numOfLayers << std::endl;
    for (int i = 0; i < numOfLayers; i++)
        std::cout<< "Number of Neurons in Layer " << i+1 << ": " << numOfNeurons[i] << std::endl;
    std::cout<< "Use the default setting? (Y/N):";
    std::cin >> defaultSetting;
    
    if (defaultSetting == "Y" || defaultSetting == "y"){
        std::cout << "Default setting chose. Press any key to continue.\n" << std::endl;
        getchar();
    }
    else{
        numOfNeurons.clear();
        
        std::cout<< "Please enter number of layers (input layer included): ";
        std::cin >> numOfLayers;
        for (int i = 1; i <= numOfLayers; i++){
            std::cout<< "Please enter number of neurons for layer " << i << ": ";
            std::cin >> userInput;
            numOfNeurons.push_back(userInput);
        }
    }
    
    //write setting to config file
    config.open(config_file.c_str(), std::ios::out);
    config << numOfLayers << std::endl;
    for (int i = 0; i < numOfNeurons.size(); i++)
        config << numOfNeurons[i] << std::endl;
    config.close();
}

int main(){
    //default setting
    //single hidden layer and 128 neurons
    numOfLayers = 3;
    numOfNeurons.resize(numOfLayers);
    numOfNeurons[0] = IN_SIZE;
    numOfNeurons[1] = HL_SIZE;
    numOfNeurons[2] = OUT_SIZE;
    
    networkConstruction();
    //============Create Network====================
	Ann training_nn(sigmoidFunct, numOfLayers, numOfNeurons, training);
    //Ann training_nn (reluFunct, numOfLayers, numOfNeurons, training);
    training_nn.chooseDataset(dataset);
    training_nn.initAnn(); 
	training_nn.summary();        //print basic network information
	training_nn.readHeader();   //get rid of header from image file (for MNIST)
	int SampleSize = training_nn.getTrainSize();
	//================Traing========================
    for (int sample = 0; sample < SampleSize; sample++) {
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
        if (sample != 0 && sample % 100 == 0) {
            std::cout << "Saving weight matrix.\n";
            training_nn.writeWeight();
        }
    }
    training_nn.writeWeight();
    std::cout << "Training Completed.\n";
    return 0;
}