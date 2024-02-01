using ActivationFunctions;
using MatrixSpace;
using LossFunctionsSpace;
using Newtonsoft.Json;
using EnumConverterSpace;

namespace NeuralNetworkSpace{
    public class NeuralNetwork{

        public NeuralNetwork(){}
        public NeuralNetwork(int[,] matrixDimensions, Atype atype, Ltype ltype ){

            Length = matrixDimensions.GetLength(0);
            Layers = new Matrix[Length];
            GradLayers=new Matrix[Length];
            Bias= new Matrix[Length];
            BiasLayers = new Matrix[Length];
            Activations= new Matrix[Length+1];
            Deltas= new Matrix[Length];

            ActivationFunctionType = atype;
            LossFunctionType = ltype;
            ActivationFunction = EnumConverter.EnumToActivation(atype);
            LossFunction = EnumConverter.EnumToLoss(ltype);

            for(int i=0;i<matrixDimensions.GetLength(0);i++){
                Layers[i] = new Matrix(matrixDimensions[i,0],matrixDimensions[i,1]);
                Bias[i]= new Matrix(matrixDimensions[i,0],1);
                GradLayers[i] = new Matrix(false,Layers[i].Dim1,Layers[i].Dim2);
                BiasLayers[i] = new Matrix(false,Layers[i].Dim1,1);

            }
        }
        public Matrix[] Layers{get;set;}

        public Random rand = new Random();

        public Matrix[] Bias{get;set;}
        [JsonIgnore] public Matrix[] Activations{get;set;}

        [JsonIgnore] public Matrix[] Deltas{get;set;}

        [JsonIgnore] public Matrix[] GradLayers{get;set;}

        [JsonIgnore] public Matrix[] BiasLayers{get;set;}

        [JsonIgnore] public ActivationFunction ActivationFunction{get; set;}  
        [JsonIgnore] public ILossFunction LossFunction{get;set;}

        public Atype ActivationFunctionType{get;set;}

        public Ltype LossFunctionType{get;set;}

        [JsonIgnore] private double Loss{get;set;}

        public int Length{get;set;}

        public Matrix Run(double[] input){
            Matrix x = new Matrix(input);
            x=ActivationFunction.ActivateFunctionDerivative(x);
            for(int i=0;i<Length;i++){
                x=ActivationFunction.ActivateFunctionDerivative(Layers[i].Multiply(x).Add(Bias[i]));
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


        private void FeedForward(double[] input){
            Matrix x= new Matrix(input);
            Activations[0]= ActivationFunction.ActivateFunction(x);
            for(int i=1;i<Length+1;i++){
                Activations[i]=ActivationFunction.ActivateFunction(Layers[i-1].Multiply(Activations[i-1]).Add(Bias[i-1]));
                

            }

        }

        private void Backpropogate(double[] output){
            var aim= new Matrix(output);   
            Deltas[Length-1]=ActivationFunction.ActivateFunctionDerivative(Activations[Length]).HadamardProduct
            (LossFunction.LossFunctionDerivative(Activations[Length],aim));
            for(int i=1;i<Length;i++){
                Deltas[Length-1-i]=ActivationFunction.ActivateFunctionDerivative(Activations[Length]).HadamardProduct(Layers[Length-i].Transpose().Multiply(Deltas[Length-i]));
            
            }
            for(int i=0;i<Length;i++){
                GradLayers[i]=GradLayers[i].Add(Deltas[i].Multiply(Activations[i].Transpose()).ScalarMultiply(0.005));
                BiasLayers[i]=BiasLayers[i].Add(Deltas[i]);
               

            }
        }

        private int[] RandomChoice(int n){
            int[] integerArray = Enumerable.Range(0,n).ToArray();
            rand.Shuffle(integerArray);
            return integerArray;

        }
        




        

    }
}