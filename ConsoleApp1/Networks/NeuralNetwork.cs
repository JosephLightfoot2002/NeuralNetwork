using System.ComponentModel.DataAnnotations;
using System.Data;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.InteropServices;
using System.Security.Principal;
using ActivationFunctions;
using Functions;
using MatrixSpace;
using NetworkEnums;
using LossFunctions;

namespace NeuralNetworkSpace{
    public abstract class NeuralNetwork{


        public Matrix[] Layers{get;set;}

        public Random rand = new Random();

        public Matrix[] Bias{get;set;}
        public Matrix[] Activations{get;set;}

        public Matrix[] Deltas{get;set;}

        public Matrix[] GradLayers{get;set;}

        public Matrix[] BiasLayers{get;set;}

        public IActivationFunction ActivationFunction{get; set;}  
        public ILossFunction LossFunction{get;set;}

        private double Loss{get;set;}

        public int Length{get;set;}

        public abstract void Result(double[][] testData);
        public Matrix Run(double[] input){
            Matrix x = new Matrix(input);
            x=ActivationFunction.ActivationFunctionDerivative(x);
            for(int i=0;i<Length;i++){
                x=ActivationFunction.ActivationFunctionDerivative(Layers[i].Multiply(x).Add(Bias[i]));
            }
            return x;
            
        }
        public void TrainNetwork(double[][] dataX, double[][] dataY, int trainTime, double testTrainSplit, int stochasticSize){

            if(testTrainSplit>1|testTrainSplit<0){
                throw new Exception("TestTrainSplit must be between 0 to 100");
            }else if(dataX.Length!=dataY.Length){
                throw new Exception("Output and Input must be same size array");
            }else if(trainTime<1){
                throw new Exception("TrainTime must be larger than 0");
            }else if(stochasticSize>dataX.Length*testTrainSplit){
                throw new Exception("Cannot randomly sample more points than in the training data");
            }else{
                
                int[] randomSplit = RandomChoice(dataX.Length);
                int trainAmount = (int)Math.Floor(testTrainSplit*dataX.Length);

                double[][] trainDataX = new double[trainAmount][];
                double[][] trainDataY = new double[trainAmount][];
                double[][] testDataX = new double[dataX.Length-trainAmount][];
                double[][] testDataY = new double[dataX.Length-trainAmount][];

                for(int i=0;i<trainAmount;i++){
                    trainDataX[i] = dataX[randomSplit[i]];
                    trainDataY[i] = dataY[randomSplit[i]];

                }
                for(int i=trainAmount;i<dataX.Length;i++){
                    testDataX[i-trainAmount] = dataX[randomSplit[i]];
                    testDataY[i-trainAmount] = dataY[randomSplit[i]];
                }

                for(int i=0; i<trainTime;i++){
                    int[] stochasticSubSet = RandomChoice(trainAmount);
                    TrainOnceNetwork(trainDataX,trainDataY,stochasticSubSet,stochasticSize);
                }


            }


        }
        public void TrainOnceNetwork(double[][] input, double[][] output, int[] stochasticChoices, int stochasticSize){
            for(int i=0;i<stochasticSize;i++){

                FeedForward(input[stochasticChoices[i]]);
                Backpropogate(output[stochasticChoices[i]]);

            }
            double N=input.Length;
            for(int i=0;i<Length;i++){
                Layers[i]=Layers[i].Add(GradLayers[i].ScalarMultiply(-1/N));
                Bias[i]=Bias[i].Add(BiasLayers[i].ScalarMultiply(-1/N));
                
                GradLayers[i]=GradLayers[i].ScalarMultiply(0);
                BiasLayers[i]=BiasLayers[i].ScalarMultiply(0);
            }

            
        }


        public void FeedForward(double[] input){
            Matrix x= new Matrix(input);
            Activations[0]= ActivationFunction.ActivationFunction(x);
            for(int i=1;i<Length+1;i++){
                Activations[i]=ActivationFunction.ActivationFunction(Layers[i-1].Multiply(Activations[i-1]).Add(Bias[i-1]));
                

            }

        }

        public void Backpropogate(double[] output){
            var aim= new Matrix(output);   
            Deltas[Length-1]=ActivationFunction.ActivationFunctionDerivative(Activations[Length]).HadamardProduct
            (LossFunction.LossFunctionDerivative(Activations[Length],aim));
            for(int i=1;i<Length;i++){
                Deltas[Length-1-i]=ActivationFunction.ActivationFunctionDerivative(Activations[Length]).HadamardProduct(Layers[Length-i].Transpose().Multiply(Deltas[Length-i]));
            
            }
            for(int i=0;i<Length;i++){
                GradLayers[i]=GradLayers[i].Add(Deltas[i].Multiply(Activations[i].Transpose()).ScalarMultiply(0.005));
                BiasLayers[i]=BiasLayers[i].Add(Deltas[i]);
               

            }
        }

        public int[] RandomChoice(int n){
            int[] integerArray = Enumerable.Range(0,n).ToArray();
            rand.Shuffle(integerArray);
            return integerArray;

        }
        




        

    }
}