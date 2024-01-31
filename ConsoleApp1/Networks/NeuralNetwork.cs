using System.ComponentModel.DataAnnotations;
using System.Data;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;
using System.Reflection.Metadata.Ecma335;
using System.Security.Principal;
using Functions;
using MatrixSpace;
using NetworkEnums;

namespace NeuralNetworkSpace{
    public abstract class NeuralNetwork{


        public required Matrix[] Layers{get;set;}

        public Matrix[] Bias{get;set;}
        public Matrix[] Activations{get;set;}

        public Matrix[] Deltas{get;set;}

        public Matrix[] GradLayers{get;set;}

        public Matrix[] BiasLayers{get;set;}

        private abstract ActivationFunctions ActivationFunction;

        private ActivationFunctionsDerivatives ActivationFunctionDerivate;
        private LossFunctions LossFunction;

        private LossFunctionDerivatives LossFunctionDerivative;
        public int Length{get;set;}

        public Matrix Run(double[] input){
            Matrix x = new Matrix(input);
            x=Activate(x);
            for(int i=0;i<Length;i++){
                x=Activate(Layers[i].Multiply(x).Add(Bias[i]));
            }
            return x;
            
        }

        public void TrainNetwork(double[][] input, double[][] output){
            for(int i=0;i<input.Length;i++){

                FeedForward(input[i]);
                Backpropogate(output[i]);

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
            Activations[0]= Activate(x);
            for(int i=1;i<Length+1;i++){
                Activations[i]=Activate(Layers[i-1].Multiply(Activations[i-1]).Add(Bias[i-1]));
                

            }

        }

        public void Backpropogate(double[] output){
            var aim= new Matrix(output);   
            Deltas[Length-1]=ActivationFunctionDerivate(Activations[Length]).HadamardProduct(Activations[Length].Add(aim.ScalarMultiply(-1)));
            for(int i=1;i<Length;i++){
                Deltas[Length-1-i]=ActivationFunctionDerivate(Activations[Length]).HadamardProduct(Layers[Length-i].Transpose().Multiply(Deltas[Length-i]));
            
            }
            for(int i=0;i<Length;i++){
                GradLayers[i]=GradLayers[i].Add(Deltas[i].Multiply(Activations[i].Transpose()));
                BiasLayers[i]=BiasLayers[i].Add(Deltas[i]);
               

            }
        }




        public Matrix Activate(Matrix m){
            return m.ApplyFunction(ActivationFunction);
        }
        

    }
}