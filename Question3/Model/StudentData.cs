using Microsoft.ML.Data;

namespace Question3.Model
{
    internal class StudentData
    {
        [LoadColumn(0)] public float STG;
        [LoadColumn(1)] public float SCG;
        [LoadColumn(2)] public float STR;
        [LoadColumn(3)] public float LPR;
        [LoadColumn(4)] public float PEG;
        [LoadColumn(5)] public string UNS;
        [LoadColumn(5), ColumnName("Label")] public float UNSNumeric ; 
    }
}
