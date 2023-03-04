package example;

import java.util.ArrayList;
import java.util.Objects;

public class Main {
    //Change values of the variables based on the type of the algorithm
    final static int numberOfIterations = 547; // OneVsOne = 547, OneVsRest = 1000
    final static double learningRate = 0.01; // OneVsOne = 0.01, OneVsRest = 0.001
    final static double lambdaParameter = 0.0005; // OneVsOne = 0.0005, OneVsRest = 0.01
    final static int numberOfClassifications = 10;

    public static void main(String[] args) {

        FileInterpreter dataSet1 = new FileInterpreter("cw2DataSet1.csv");
        FileInterpreter dataSet2 = new FileInterpreter("cw2DataSet2.csv");

//      Svm classifier one to one
        SVMMultipleClassifierOneToOne svm = new SVMMultipleClassifierOneToOne(learningRate, lambdaParameter, numberOfIterations, numberOfClassifications);
        svm.fit(dataSet1.getData().inputs, dataSet1.getData().targets);
        ArrayList<Integer> predictions = svm.predict(dataSet2.getData().inputs);

//      Svm classifier one to rest
//        SVMMultipleClassifierOneToRest svm = new SVMMultipleClassifierOneToRest(numberOfIterations, numberOfClassifications, learningRate, lambdaParameter);
//        svm.fit(dataSet1.getData().inputs, dataSet1.getData().targets);
//        ArrayList<Integer> predictions = svm.predict(dataSet2.getData().inputs);

        int count = 0;
        for (int i = 0; i < predictions.size(); i++) {

            if (Objects.equals(predictions.get(i), dataSet2.getData().targets.get(i))) count++;
        }

        double accuracy = (count / (double) predictions.size()) * 100;
        System.out.println(count + "/" + predictions.size() + " = " + accuracy + "%");
    }
}
