using MatrixSpace;
namespace Functions{

    public delegate double ActivationFunctions(double input);

    public delegate double LossFunctions(Matrix x, Matrix y);

    public delegate Matrix ActivationFunctionsDerivatives(Matrix matrix);
    public delegate Matrix LossFunctionDerivatives(Matrix x,Matrix y);




}