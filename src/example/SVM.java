package example;

import java.util.ArrayList;
import java.util.Arrays;

public class SVM {


    private final double learningRate;
    private final double lambdaParam;
    private final int numberOfIterations;
    private double[] weights;
    private double bias;
    private int[] classMap;

    public SVM(double learningRate, double lambdaParam, int numberOfIterations) {
        this.learningRate = learningRate;
        this.lambdaParam = lambdaParam;
        this.numberOfIterations = numberOfIterations;
    }

    private void initWeightsAndBias(ArrayList<ArrayList<Integer>> inputData){

        int numberOfFeatures = inputData.get(0).size();
        this.weights = new double[numberOfFeatures];
        this.bias = 0;
    }

    private int[] getClassMap(ArrayList<Integer> labelData){
        final int[] result = new int[labelData.size()];

        for (int i = 0; i < labelData.size(); i++) {
            if(labelData.get(i) <= 0)result[i] = -1;
            else result[i] = 1;
        }
        return result;
    }

    private boolean satisfyConstraint(ArrayList<Integer> inputFeature, int inputFeatureId){


       final double linearModel = getSumOfTwoArrays(inputFeature, this.weights) + this.bias;
//        this.error = sigmoid(linearModel);
        return this.classMap[inputFeatureId] * linearModel >= 1;
    }


    private double sigmoid(double number){
        return 1 /(1 + Math.pow(Math.E, -number));
    }

    private double relu(double number){
        return number > 0 ? number : (0.01D *  number);
    }

    private double getSumOfTwoArrays(ArrayList<Integer> inputFeature, double[] weights){
        double sum = 0;

        for (int i = 0; i < inputFeature.size(); i++) {
            sum += inputFeature.get(i) * weights[i];
        }
        return sum;
    }

    private Gradients getGradients(boolean constrain, ArrayList<Integer> inputFeature, int inputFeatureId){

        Gradients gradients = new Gradients();

         if(constrain){
         gradients.dWeights = Arrays.stream(this.weights).map(number -> this.lambdaParam * number).toArray();
         gradients.dBias = 0;
         }else{
            gradients.dWeights = getNewWeights(inputFeature, inputFeatureId);
         }
        return gradients;
    }

    private double[] getNewWeights(ArrayList<Integer> inputFeature, int inputFeatureId){
        final double[] result = new double[inputFeature.size()];

        for (int i = 0; i < result.length; i++) {
            result[i] = this.lambdaParam * (this.weights[i] - (inputFeature.get(i) * this.classMap[inputFeatureId]));
        }
        return result;
    }

    private void updateWeightsAndBias(Gradients gradients){
        final double[] newWeights = new double[gradients.dWeights.length];

        for (int i = 0; i < newWeights.length; i++) {
            newWeights[i] = this.weights[i] - (this.learningRate * gradients.dWeights[i]);
        }

        this.weights = newWeights;
        this.bias -= this.learningRate * gradients.dBias;
    }

    public void fit(ArrayList<ArrayList<Integer>> inputData, ArrayList<Integer> labelData){

        initWeightsAndBias(inputData);
        this.classMap = getClassMap(labelData);

        for (int i = 0; i < this.numberOfIterations; i++) {

            for (int j = 0; j < inputData.size(); j++) {

                final boolean constrain = satisfyConstraint(inputData.get(j),j);
                final Gradients gradients = getGradients(constrain,inputData.get(j),j);

                updateWeightsAndBias(gradients);
            }
        }
    }

    public int[] predict(ArrayList<ArrayList<Integer>> inputData){
        final double[] estimate = getEstimate(inputData);
        return getPrediction(estimate);
    }

    private double[] getEstimate(ArrayList<ArrayList<Integer>> inputData){
        final double[] result = new double[inputData.size()];

        for (int i = 0; i < inputData.size(); i++) {
            double sum = 0;
            for (int j = 0; j < inputData.get(0).size(); j++) {
                sum+= inputData.get(i).get(j) * this.weights[j];
            }
            result[i] = sum + this.bias;
        }
        return result;
    }

    private int[] getPrediction(double[] estimate){

        final int[] result = new int[estimate.length];

        for (int i = 0; i < result.length; i++) {
            if(estimate[i] <= 0) result[i] = -1;
            else result[i] = 1;
        }

        return result;
    }
}
