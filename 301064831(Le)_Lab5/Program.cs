using System;
using System.IO;
using _301064831_Le__Lab5.Question_2;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace _301064831_Le__Lab5
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Defining ML context
            MLContext mLContext = new MLContext();

            // Load the data
            var dataPath = Path.Combine(Environment.CurrentDirectory, "insurance.csv");
            var data = mLContext.Data.LoadFromTextFile<InsuranceData>(dataPath, separatorChar: ',', hasHeader: true);

            // Split the data into training and testing sets
            var trainingData = mLContext.Data.TrainTestSplit(data);

            // Define the pipeline
            var pipeline = mLContext.Transforms.CopyColumns("Label", "Charges")
                .Append(mLContext.Transforms.Categorical.OneHotEncoding("Sex"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding("Smoker"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding("Region"))
                .Append(mLContext.Transforms.Conversion.ConvertType("Children", outputKind: DataKind.Single))
                .Append(mLContext.Transforms.Concatenate("Features", "Age", "Sex", "BMI", "Children", "Smoker", "Region"))
                .Append(mLContext.Regression.Trainers.Sdca(labelColumnName: "Charges"));

            // Training model
            var model = pipeline.Fit(trainingData.TrainSet);

            // Make predictions on the test set
            var predictions = model.Transform(trainingData.TestSet);


            // Making a prediction using the provided data.
            var predictionEngine = mLContext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
            var newData = new InsuranceData
            {
                Age = 65,
                Sex = "female",
                BMI = 21,
                Children = 3,
                Smoker = "yes",
                Region = "northeast"
            };

            var prediction = predictionEngine.Predict(newData);
            Console.WriteLine($"Predicted Medical Cost: {prediction.Charges:C}");
        }
    }
}
