
using System.Diagnostics.CodeAnalysis;
using System.Security.Claims;
using System.Security.Cryptography.X509Certificates;
using System.Text.Json.Serialization;
using ActivationFunctions;
namespace MatrixSpace{

    public class Matrix{

        public Matrix(){}
        public Matrix(double[,] matrix){
            Values=matrix;
            Dim1=matrix.GetLength(0);
            Dim2=matrix.GetLength(1);

        }
        public Matrix(double[] vector){
            Values= new double[vector.Length,1];
            Dim1=vector.Length;
            Dim2=1;
            for(int i=0;i<vector.Length;i++){
                Values[i,0]=vector[i];

            }
        }

        public Matrix(int dim1, int dim2){
            Random random = new Random();
            Values=new double[dim1,dim2];
            double d=dim2;
            for(int i=0;i<dim1;i++){
                for(int j=0;j<dim2;j++){
                    Values[i,j]=RandomMethods.GenerateRN(random,0,1/d);
                }
            }
            Dim1=dim1;
            Dim2=dim2;

        }
        public Matrix(bool identity,int dim1,int dim2){
            Dim1=dim1;
            Dim2=dim2;
            Values= new double[dim1,dim2];
            if(identity){
                for(int i=0;i<dim1;i++){
                    for(int j=0;j<dim2;j++){
                        Values[i,j]=1;  
                    }
                
                }
            }else{
                for(int i=0;i<dim1;i++){
                    for(int j=0;j<dim2;j++){
                        Values[i,j]=0;  
                    }
                }
            }


        }

        public double[,] Values{get;set;}

        public int Dim1{get;set;}

        public int Dim2{get;set;}


        public Matrix Multiply(Matrix m2){
            double[,] values= new double[Dim1,m2.Dim2];
            for(int i=0;i<Dim1;i++){
                for(int j=0;j<m2.Dim2;j++){
                    double sum=0;
                    for(int k=0;k<Dim2;k++){
                        sum+=Values[i,k]*m2.Values[k,j];
                    }
                    values[i,j]=sum;
                }
            }
            return new Matrix(values);

        }

        public Matrix HadamardProduct(Matrix m){
            double[,] values= new double[Dim1,Dim2];
            for(int i=0;i<Dim1;i++){
                for(int j=0;j<Dim2;j++){
                    values[i,j]=m.Values[i,j]*Values[i,j];
                }

            }
            return new Matrix(values);

        }

        public Matrix Add(Matrix m){
            double[,] values = new double[Dim1,Dim2];
            for(int i=0;i<Dim1;i++){
                for(int j=0;j<Dim2;j++){
                    values[i,j]=Values[i,j]+m.Values[i,j];


                }
            }

            return new Matrix(values);
        }

        public double Point(int i, int j){
            return Values[i,j];
        }

        public Matrix ScalarMultiply(double scalar){
            double[,] values= new double[Dim1,Dim2];
            for(int i=0;i<Dim1;i++){
                for(int j=0;j<Dim2;j++){
                    values[i,j]=scalar*Values[i,j];

                }
            }
            return new Matrix(values);
        }
        
        public Matrix ScalarAdd(double scalar){
            double[,] values=new double[Dim1,Dim2];
            for(int i=0;i<Dim1;i++){
                for(int j=0;j<Dim2;j++){
                    values[i,j]= Values[i,j]+scalar;
                }
            }
            return new Matrix(values);
        }
        

        public Matrix Transpose(){
            double[,] values = new double[Dim2,Dim1];
            for(int i=0;i<Dim1;i++){
                for(int j=0;j<Dim2;j++){
                    values[j,i]=Values[i,j];
                }
            }

            return new Matrix(values);
        }



        public void PrintMatrix(){
            string s= new string("");
            for(int i=0;i<Dim1;i++){
                for(int j=0;j<Dim2;j++){
                    if(j==0){
                        s=s+"[ "+Values[i,j].ToString();

                    }else if(j==Dim2-1){
                        s=s+Values[i,j].ToString()+"]\n";
                    }else{
                        s=s+Values[i,j].ToString()+" ";
                    }

                }
            }
            Console.WriteLine(s);
        }

        public double Norm(){
            double sum=0;
            for(int i=0;i<Dim1;i++){
                for(int j=0;j<Dim2;j++){
                    sum+=Math.Pow(Values[i,j],2)
;
                }
            }
            return sum;
        }

        public Matrix ApplyFunction(Function f){
            double[,] values= new double[Dim1,Dim2];
            for(int i=0;i<Dim1;i++){
                for(int j=0;j<Dim2;j++){
                    values[i,j]=f(Values[i,j]);
                }
            }
            return new Matrix(values);
        }

        public Matrix VectorToMatrix(){
            if(Dim2>1){
                throw new ArgumentException("Not a Vector");
            }else{
                Matrix matrix =  new Matrix(true,Dim1,Dim1);
                for(int i=0;i<Dim1; i++){
                    matrix.Values[i,i] = Values[i,1];
                }
                return matrix;
            }
            
        }

    }
}