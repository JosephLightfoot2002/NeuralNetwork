
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Security.Cryptography.X509Certificates;
using MatrixSpace;

namespace ActivationFunctions{



    public delegate double Function(double x);
    public interface IActivationFunction{
        Matrix ActivationFunction(Matrix x);
        Matrix ActivationFunctionDerivative(Matrix x);

    }

    public class SigmoidalFunction:IActivationFunction{

        public Function f = new Function(x=>1/(1+Math.Exp(-x)));
        public Function g = new Function(x=>x*1-x);
        public Matrix ActivationFunction(Matrix x){
            return x.ApplyFunction(f);
        }
        public Matrix ActivationFunctionDerivative(Matrix x){
            return x.ApplyFunction(g);

        }


    }

    public class ReLu:IActivationFunction{

        Function f = new Function(x=>Math.Max(0,x));
        Function g = new Function(x=>{if(x>0){return 1;}else{return 0;}});
        public Matrix ActivationFunction(Matrix x){
            return x.ApplyFunction(f);
        }
        
        public Matrix ActivationFunctionDerivative(Matrix x){
            return x.ApplyFunction(g);
        }
    }

    public class SoftMax:IActivationFunction{

        Function f= new Function(x=>Math.Exp(x/2));
        public Matrix ActivationFunction(Matrix x ){
            x.ApplyFunction(f);
            double tot=x.Norm();
            x.ApplyFunction(x=>x/tot);
            return x;
        }

        public Matrix ActivationFunctionDerivative( Matrix x){
            return x;
        }
    }

    public class Linear:IActivationFunction{
        public Matrix ActivationFunction(Matrix x){
            return x;
        }

        public Matrix ActivationFunctionDerivative(Matrix x){
            return new Matrix(true, x.Dim1,x.Dim2);
        }
    }

}