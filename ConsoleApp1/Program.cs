// See https://aka.ms/new-console-template for more information
using System.Security.Cryptography.X509Certificates;
using MatrixSpace;
using NeuralNetworkSpace;
using MSENeuralNetworkSpace;
using ActivationFunctions;
using NNJsonConverterSpace;

int[,] dim={{1,2}};
ActivationFunction f = new Linear();
var hi = new MSENeuralNetwork(dim,Atype.Linear);
double[][] x= {new double[]{1,3}, new double[]{2,7},new double[]{3,8},new double[]{4,-4},new double[]{5,-16},new double[]{6,22}};
double[][] y ={new double[]{3}, new double[]{8},new double[]{10},new double[]{-1},new double[]{-12},new double[]{27}};

hi.TrainNetwork(x,y,2000,0.8,2);
Console.WriteLine(hi.Run(x[0]).ToString());
string j = NNJsonConverter.NNtoJson(hi);
var hi2 = NNJsonConverter.JsontoNN(j);
Console.WriteLine("hi");

// double[][] trainingInput = [[0,0],[1,0],[0,1],[1,1]];
// Random rnd= new Random();


// double[][] trainingOutput = [[0],[0],[0],[1]];

// var hi=CsvReader.Read();

// int[,] dims = {{10,hi[0][0].Length}};
// NeuralNetwork  neural= new NeuralNetwork(dims);
// for(int i=0;i<10000;i++){
//     var sample= RandomMethods.StochasticChoice(hi,150);
//     double[][] trainingInput=sample[0];
//     double[][] trainingOutput=sample[1];
//     neural.TrainNetwork(trainingInput,trainingOutput);
//     if(i%10==0){
//         Console.WriteLine(i);
//     }
// }
// new Matrix(hi[1][0]).PrintMatrix();
// neural.Run(hi[0][0]).PrintMatrix();
// var hi=neural.Run(trainingInput[0]);
// hi.PrintMatrix();
// hi=neural.Run(trainingInput[1]);
// hi.PrintMatrix();
// hi=neural.Run(trainingInput[2]);
// hi.PrintMatrix();
// hi=neural.Run(trainingInput[3]);
// hi.PrintMatrix();

