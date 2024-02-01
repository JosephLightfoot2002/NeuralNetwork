using System;

public static class RandomMethods{
    public static double GenerateRN(Random random, double mean, double stdDev){
        double u1= 1-random.NextDouble();
        double u2 = 1-random.NextDouble();
        double z0=Math.Sqrt(-2*Math.Log(u1))+Math.Cos(2*Math.PI*u2);
        var x=mean+stdDev*z0;
        return x;
    }

    public static List<double[][]> StochasticChoice(List<double[][]> l, int n){
        Random rnd = new Random();
        var returnList = new List<double[][]>();
        var trainingI= new double[n][];
        var trainingO= new double[n][];
        List<int> repeats= new List<int>();
        for(int i=0; i<n;i++){
            int place= rnd.Next(0,60000);
            while(repeats.Any(x=>x==place)){
                place=rnd.Next(0,60000);
                
            }
            trainingI[i]=l[0][place];
            trainingO[i]=l[1][place];
            repeats.Add(place);

        }
        returnList.Add(trainingI);
        returnList.Add(trainingO);
        return returnList;


    }
}