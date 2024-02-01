
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Security.Cryptography.X509Certificates;
using MatrixSpace;

namespace ActivationFunctions{



    public delegate double Function(double x);
    public abstract class ActivationFunction{

        protected Function F{get;}
        protected Function dF{get;}
        public virtual Matrix ActivateFunction(Matrix x){
            return x.ApplyFunction(F);
        }
        public virtual Matrix ActivateFunctionDerivative(Matrix x){
            return x.ApplyFunction(F);
        }

    }

    public class Sigmoidal:ActivationFunction{

        public Function f = new Function(x=>1/(1+Math.Exp(-x)));
        public Function g = new Function(x=>x*1-x);
    }

    public class ReLu:ActivationFunction{

        Function f = new Function(x=>Math.Max(0,x));
        Function g = new Function(x=>{if(x>0){return 1;}else{return 0;}});
    }

    public class SoftMax:ActivationFunction{

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

    public class Linear:ActivationFunction{
        public override Matrix ActivateFunction(Matrix x){
            return x;
        }

        public override Matrix ActivateFunctionDerivative(Matrix x){
            return new Matrix(true, x.Dim1,x.Dim2);
        }
    }

    public enum Atype{
        Linear,
        Sigmoidal,
        ReLu,
        SoftMax
    }
    
    

}