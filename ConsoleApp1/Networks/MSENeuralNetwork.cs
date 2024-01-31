using NeuralNetworkSpace;
using Functions;
using NetworkEnums;
using MatrixSpace;
namespace MSENeuralNetworkSpace{

    public class MSENeuralNetwork:NeuralNetwork{
        
        public MSENeuralNetwork(int[,] matrixDimensions, Atype aType){
            
            Length=matrixDimensions.GetLength(0);
            Layers = new Matrix[Length];
            GradLayers=new Matrix[Length];
            Bias= new Matrix[Length];
            BiasLayers = new Matrix[Length];
            Activations= new Matrix[Length+1];
            Deltas= new Matrix[Length];

            for(int i=0;i<matrixDimensions.GetLength(0);i++){
                Layers[i] = new Matrix(matrixDimensions[i,0],matrixDimensions[i,1]);
                Bias[i]= new Matrix(matrixDimensions[i,0],1);
                GradLayers[i] = new Matrix(false,Layers[i].Dim1,Layers[i].Dim2);
                BiasLayers[i] = new Matrix(false,Layers[i].Dim1,1);

            }
 
            if(aType==Atype.Sigmoidal){
                ActivationFunction=x=>1/(1+Math.Exp(-x));
                ActivationFunctionDerivate=x=>x.HadamardProduct(x.ScalarMultiply(-1).ScalarAdd(1));

            }else if(aType==Atype.ReLu){
                ActivationFunction=x=>Math.Max(x,0);
                ActivationFunctionDerivate=x=>x.ReLuFunc();

            }else{
                
            }
            if(lType==LossType.MSE){
                LossFunction=(x,y)=>{
                    return 0.5*x.Add(y.ScalarMultiply(-1)).Norm();
                };
                LossFunctionDerivative=(x,y)=>{
                    return x.Add(y.ScalarMultiply(-1));
                };

            }else if(lType==LossType.Entropy){
                LossFunction=(x,y)=>{
                    return -x.Transpose().Multiply(y.ApplyFunction(x=>Math.Log(x))).Point(0,0);
                };
                LossFunctionDerivative=(x,y)=>{
                    x
                    
                };
            }

        }
    }


}