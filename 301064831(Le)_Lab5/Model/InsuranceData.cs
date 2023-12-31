using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace _301064831_Le__Lab5.Question_2
{
    internal class InsuranceData
    {
        [LoadColumn(0)] public float Age;
        [LoadColumn(1)] public string Sex;
        [LoadColumn(2)] public float BMI;
        [LoadColumn(3)] public int Children;
        [LoadColumn(4)] public string Smoker;
        [LoadColumn(5)] public string Region;
        [LoadColumn(6)] public float Charges;
    }
}
