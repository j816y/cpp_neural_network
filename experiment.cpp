#include "ann.h"

//3D vector to store all weight in neural network
// vector[layer_index][neuron_index][weight_index]
std::vector<std::vector<std::vector<float> > > weight;
std::vector<std::vector<std::vector<float> > >* wPtr;

int userInput;
int numOfLayers;

//number of neuron for each layer, vector size based on number of layers
std::vector<int> numOfNeurons;

void init(){
    std::cout<< "Please enter number of layers (input layer included): ";
    std::cin >> numOfLayers;
    for (int i = 1; i <= numOfLayers; i++){
        std::cout<< "Please enter number of neurons for layer " << i << ": ";
        std::cin >> userInput;
        numOfNeurons.push_back(userInput);
    }
    for (int i = 0; i < numOfNeurons.size(); i++)
        std::cout << "Layer " << i+1 << " contains " << numOfNeurons[i] << " neurons.\n";
    
    int interconnect = numOfLayers - 1; //number of sets of connections between each layer. For a 3 layers network there are 2 sets of interconnects in between
    //weight initialization
    weight.resize(interconnect);
    for (int i = 0; i < interconnect; i++){
        std::cout << "Interconnect " << i+1 << std::endl;
        //weight stored in the following format: w[numOfneuron][numOfInput]
        weight[i].resize(numOfNeurons[i+1]);
        for (int j = 0; j < numOfNeurons[i+1]; j++){
            std::cout << "  Neuron " << j+1 << std::endl;
            for (int k = 0; k < numOfNeurons[i]; k++){
                weight[i][j].push_back(k);   
            }
            std::cout << "      Number of Weight " << weight[i][j].size() << std::endl;
        }
    }
}

int main(){
    // init();
    // wPtr = &weight;
    // std::cout << "Sets of Interconnects:    " << (*wPtr).size() << std::endl;
    // std::cout << "Number of Weights:    " << (*wPtr)[0].size() << std::endl;
    // std::cout << "Sets of Interconnects:    " << (*wPtr)[0][0].size() << std::endl;
    // std::cout << "Number of Weights:    :    " << (*wPtr)[1].size() << std::endl;
    // std::cout << "Sets of Interconnects:    " << (*wPtr)[1][0].size() << std::endl;
    for (int a = 10-1; a >= 0; a--)
        std::cout << a << std::endl;
    return 0;
}