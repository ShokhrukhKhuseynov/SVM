package example;

import java.util.ArrayList;


public class SVMMultipleClassifierOneToOne {

    final int numberOfClassifications;
    final ArrayList<ArrayList<SVM>> vectors;

    public SVMMultipleClassifierOneToOne(double learningRate, double lambdaParam, int numberOfIterations, int numberOfClassifications) {
        this.numberOfClassifications = numberOfClassifications;
        this.vectors = new ArrayList<>();
        for (int i = 0; i < numberOfClassifications; i++) {
            this.vectors.add(new ArrayList<>());
            for (int j = 0; j < numberOfClassifications; j++) {
                this.vectors.get(i).add(new SVM(learningRate,lambdaParam,numberOfIterations));
            }
        }
    }

    public void fit(ArrayList<ArrayList<Integer>> inputData, ArrayList<Integer> labelData){

        for (int i = 0; i < this.numberOfClassifications; i++) {

            for (int j = 0; j < this.numberOfClassifications; j++) {

                if(i != j){

                    final DataReturner data = getFormattedLabelDataByClass(inputData,labelData,i,j);
                    this.vectors.get(i).get(j).fit(data.inputs, data.targets);
                }
            }
        }
    }

    private DataReturner getFormattedLabelDataByClass(ArrayList<ArrayList<Integer>> inputData, ArrayList<Integer> labelData, int classType1, int classType2){
        final DataReturner result = new DataReturner();

        for (int i = 0; i < inputData.size(); i++) {

            if (labelData.get(i) == classType1 || labelData.get(i) == classType2){

                result.inputs.add(inputData.get(i));
                if(labelData.get(i) == classType1) result.targets.add(0);
                else if(labelData.get(i) == classType2) result.targets.add(1);
            }
        }
        return result;
    }

    public ArrayList<Integer> predict(ArrayList<ArrayList<Integer>> inputData){

        final ArrayList<Integer> result = new ArrayList<>();

        for (ArrayList<Integer> inputDatum : inputData) {
            final ArrayList<ArrayList<Integer>> outerArray = new ArrayList<>();

            for (int j = 0; j < this.numberOfClassifications; j++) {
                final ArrayList<Integer> innerArray = new ArrayList<>();
                for (int k = 0; k < this.numberOfClassifications; k++) {

                    if (j != k) {
                        final ArrayList<ArrayList<Integer>> input = new ArrayList<>();
                        input.add(inputDatum);
                        final int prediction = this.vectors.get(j).get(k).predict(input)[0];
                        if (prediction == 1) break;
                        else innerArray.add(j);
                    }
                }
                outerArray.add(innerArray);
            }
            ArrayList<Integer> output = new ArrayList<>(0);
            for (ArrayList<Integer> array : outerArray) {
                if (output.size() < array.size()) output = array;
            }
            result.add(output.get(0));
        }
        return result;
    }

}
