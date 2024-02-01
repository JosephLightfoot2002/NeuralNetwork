
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using MatrixSpace;

namespace LossFunctionsSpace{



    public delegate double Function(double x);
    public interface ILossFunction{
        double LossFunction(Matrix x, Matrix y);
        Matrix LossFunctionDerivative(Matrix x, Matrix y);

    }

    public class MeanSquareError:ILossFunction{

        public double LossFunction(Matrix x, Matrix y){
            return x.Add(y.ScalarMultiply(-1)).Norm();
        }
        public Matrix LossFunctionDerivative(Matrix x, Matrix y){
            return x.Add(y.ScalarMultiply(-1));

        }


    }

    public class CrossEntropyLoss:ILossFunction{
        
        public double LossFunction(Matrix x, Matrix y){
            x.ApplyFunction(x=>Math.Log(x));
            return y.Transpose().Multiply(x).Point(1,1);

        }
        public Matrix LossFunctionDerivative(Matrix x, Matrix y){
            return x.Add(y.ScalarMultiply(-1));

        }
    }

    public enum Ltype{
        MeanSquareError
    }

    

}