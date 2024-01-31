// See https://aka.ms/new-console-template for more information
using System.Security.Cryptography.X509Certificates;
using MatrixSpace;
using NeuralNetworkSpace;

// double[][] trainingInput = [[0,0],[1,0],[0,1],[1,1]];
// Random rnd= new Random();


// double[][] trainingOutput = [[0],[0],[0],[1]];

var hi=CsvReader.Read();

int[,] dims = {{10,hi[0][0].Length}};
NeuralNetwork  neural= new NeuralNetwork(dims);
for(int i=0;i<10000;i++){
    var sample= RandomMethods.StochasticChoice(hi,150);
    double[][] trainingInput=sample[0];
    double[][] trainingOutput=sample[1];
    neural.TrainNetwork(trainingInput,trainingOutput);
    if(i%10==0){
        Console.WriteLine(i);
    }
}
new Matrix(hi[1][0]).PrintMatrix();
neural.Run(hi[0][0]).PrintMatrix();
// var hi=neural.Run(trainingInput[0]);
// hi.PrintMatrix();
// hi=neural.Run(trainingInput[1]);
// hi.PrintMatrix();
// hi=neural.Run(trainingInput[2]);
// hi.PrintMatrix();
// hi=neural.Run(trainingInput[3]);
// hi.PrintMatrix();

