using System;
using System.Data;
using Newtonsoft.Json;
using NeuralNetworkSpace;

namespace NNJsonConverterSpace{

    public static class NNJsonConverter{

        public static string NNtoJson(NeuralNetwork neuralNetwork){
            return JsonConvert.SerializeObject(neuralNetwork);
        }

        public static NeuralNetwork JsontoNN(string json){
            var neuralNetwork = JsonConvert.DeserializeObject<NeuralNetwork>(json);
            neuralNetwork.GenerateFromJson();
            return neuralNetwork;
        }
    }

    // public class NeuralNetworkJsonPackager(){
    //     public NeuralNetworkJsonPackager(NeuralNetwork neuralNetwork,)
    // }

}