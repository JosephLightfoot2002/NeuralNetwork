using System.Collections;
using ActivationFunctions;
using LossFunctionsSpace;
namespace EnumConverterSpace{

    public static class EnumConverter{
        public static ActivationFunction EnumToActivation(Atype atype){
            switch(atype){
                case Atype.Sigmoidal:
                    return new Sigmoidal();
                case Atype.Linear:
                    return new Linear();
                case Atype.SoftMax:
                    return new SoftMax();
                case Atype.ReLu:
                    return new ReLu();
                default:
                throw new ArgumentException("not Valid");
            }
        }

        public static ILossFunction EnumToLoss(Ltype ltype){
            switch(ltype){
                case Ltype.MeanSquareError:
                    return new MeanSquareError();
                
                default:
                    throw new ArgumentException("not Valid");
            }
        }
    }


}