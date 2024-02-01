using NeuralNetworkSpace;
using MatrixSpace;
using LossFunctionsSpace;
using ActivationFunctions;

namespace MSENeuralNetworkSpace{

    public class MSENeuralNetwork:NeuralNetwork{
        
        public MSENeuralNetwork(int[,] matrixDimensions, Atype x):base(matrixDimensions,x,Ltype.MeanSquareError){
            
 

        }
    }


}