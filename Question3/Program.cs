using Microsoft.ML;
using System;
using System.IO;
using Question3.Model;
using Microsoft.ML.Data;


namespace Question3
{
    class Program
    {
        static void Main()
        {
            // Set up the MLContext
            var mlContext = new MLContext();

            // Load the data 
            //var dataPath = "C:\\Users\\nelso\\Desktop\\Centennial College\\Semester 4\\Programming 3 (SEC. 402) - COMP 212-402\\301064831(Le)_Lab5\\Question3\\bin\\Debug\\net6.0\\Student.csv";
            var dataPath = Path.Combine(Environment.CurrentDirectory,"Student.csv");
            var data = mlContext.Data.LoadFromTextFile<StudentData>(dataPath, separatorChar: ',');

            // Define the pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(StudentData.UNS))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("Label"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("UNSNumeric", nameof(StudentData.UNS)))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("UNSNumeric"))
                .Append(mlContext.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Single)) 
                .Append(mlContext.Transforms.Concatenate("Features", "STG", "SCG","STR","LPR","PEG", "Label"))
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", maximumNumberOfIterations: 100));

            // Train the model
            var model = pipeline.Fit(data);

            // Make predictions
            var predictions = model.Transform(data);

            // Evaluate the model
            var metrics = mlContext.Regression.Evaluate(predictions, "Label");

            //Predict knowledge level for a new student
            var predictionEngine = mlContext.Model.CreatePredictionEngine<StudentData, StudentPrediction>(model);

            var newStudent = new StudentData
            {
                STG = 0.5f,
                SCG = 0.6f,
                STR = 0.7f,
                LPR = 0.8f,
                PEG = 0.9f,
                UNSNumeric = 0.0f 
            };

            var prediction = predictionEngine.Predict(newStudent);

            Console.WriteLine($"Predicted Knowledge Level: {prediction.UNS}");
        }
    }
}
