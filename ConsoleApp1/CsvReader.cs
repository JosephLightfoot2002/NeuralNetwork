using System.ComponentModel;
using System.IO;
public static class CsvReader{
    public static List<double[][]> Read()
    {
        using(var reader = new StreamReader("mnist_train.csv"))
        {
            var obj= new List<double[][]>();
            double[][] trainInput= new double[60000][];
            double[][] trainOutput= new double[60000][];
            var line=reader.ReadLine();
            int i=0;
            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();
                var values = line.Split(',');

                trainInput[i]=values.Skip(1).Select(double.Parse).ToArray();
                trainOutput[i]=IntToVector(int.Parse(values[0]));
                i++;
            }
            obj.Add(trainInput);
            obj.Add(trainOutput);
            return obj;
        }
        
    }

    public static double[] IntToVector(int i){
        double[] x ={0,0,0,0,0,0,0,0,0,0};
        x[i]=1;
        return x;

    }
}