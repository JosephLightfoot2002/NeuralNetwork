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

        public Matrix[] Bias{get;set;}
        public Matrix[] Activations{get;set;}

        public Matrix[] Deltas{get;set;}

        public Matrix[] GradLayers{get;set;}

        public Matrix[] BiasLayers{get;set;}

        public IActivationFunction ActivationFunction{get; set;}  
        public ILossFunction LossFunction{get;set;}

        public int Length{get;set;}

        public Matrix Run(double[] input){
            Matrix x = new Matrix(input);
            x=ActivationFunction.ActivationFunctionDerivative(x);
            for(int i=0;i<Length;i++){
                x=ActivationFunction.ActivationFunctionDerivative(Layers[i].Multiply(x).Add(Bias[i]));
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
            Activations[0]= ActivationFunction.ActivationFunctionDerivative(x);
            for(int i=1;i<Length+1;i++){
                Activations[i]=ActivationFunction.ActivationFunctionDerivative(Layers[i-1].Multiply(Activations[i-1]).Add(Bias[i-1]));
                

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
                GradLayers[i]=GradLayers[i].Add(Deltas[i].Multiply(Activations[i].Transpose()));
                BiasLayers[i]=BiasLayers[i].Add(Deltas[i]);
               

            }
        }




        

    }
}