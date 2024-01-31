using NeuralNetworkSpace;
using Functions;
using NetworkEnums;
using MatrixSpace;
using ActivationFunctions;
using LossFunctions;
namespace MSENeuralNetworkSpace{

    public class MSENeuralNetwork:NeuralNetwork{
        
        public MSENeuralNetwork(int[,] matrixDimensions, IActivationFunction x ){
            
            Length=matrixDimensions.GetLength(0);
            Layers = new Matrix[Length];
            GradLayers=new Matrix[Length];
            Bias= new Matrix[Length];
            BiasLayers = new Matrix[Length];
            Activations= new Matrix[Length+1];
            Deltas= new Matrix[Length];

            ActivationFunction = x;
            LossFunction = new MeanSquareError();

            for(int i=0;i<matrixDimensions.GetLength(0);i++){
                Layers[i] = new Matrix(matrixDimensions[i,0],matrixDimensions[i,1]);
                Bias[i]= new Matrix(matrixDimensions[i,0],1);
                GradLayers[i] = new Matrix(false,Layers[i].Dim1,Layers[i].Dim2);
                BiasLayers[i] = new Matrix(false,Layers[i].Dim1,1);

            }
 

        }
    }


}