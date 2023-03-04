package example;

import java.util.ArrayList;

public class SVMMultipleClassifierOneToRest {

    private final int numberOfClassifications;
    private final SVM[] vectors;
    private final double learningRate;
    private final double lambdaParam;
    private final int numIterations;


    public SVMMultipleClassifierOneToRest(int numIterations, int numberOfClassifications, double learningRate, double lambdaParam) {
        this.numberOfClassifications = numberOfClassifications;
        this.vectors = new SVM[numberOfClassifications];
        this.learningRate = learningRate;
        this.lambdaParam = lambdaParam;
        this.numIterations = numIterations;
    }

    public void fit(ArrayList<ArrayList<Integer>> inputData, ArrayList<Integer> labelData){

        for (int i = 0; i < this.numberOfClassifications ; i++) {

            final SVM svm = new SVM(this.learningRate, this.lambdaParam, this.numIterations);
            final ArrayList<Integer> formattedLabels = getFormattedLabelDataByClass(labelData, i);
            svm.fit(inputData, formattedLabels);
            this.vectors[i] = svm;
        }

    }

    public ArrayList<Integer> predict(ArrayList<ArrayList<Integer>> inputData){

        final int[][] tempResult = new int[inputData.size()][this.numberOfClassifications];

        for (int i = 0; i < inputData.size(); i++) {
            final int[] tempArray = new int[this.numberOfClassifications];

            for (int j = 0; j < this.numberOfClassifications; j++) {

                final ArrayList<ArrayList<Integer>> sample = new ArrayList<>();
                sample.add(inputData.get(i));
                tempArray[j] = this.vectors[j].predict(sample)[0];
            }
            tempResult[i] = tempArray;
        }
        ArrayList<Integer> result = new ArrayList<>();

        for (int [] array: tempResult) {
            final int res = contains(array);
            if(res != -1) result.add(res);
            else result.add((int) Math.floor(Math.random() * 10));
        }
        return result;
    }

    private int contains(int[] array){

        for (int i = 0; i < array.length; i++) {
            if (array[i] == -1) return i;
        }
        return -1;
    }
    private ArrayList<Integer> getFormattedLabelDataByClass(ArrayList<Integer> labelData, int label){

        return labelData.stream().map(number ->{
            if(number == label) return 0;
            else return 1;
        }).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
}
