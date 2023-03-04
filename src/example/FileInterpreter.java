package example;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;

public class FileInterpreter {

    private final ArrayList<ArrayList<Integer>> inputData = new ArrayList<>();
    private final ArrayList<Integer> labelData = new ArrayList<>();


    public FileInterpreter(String fileName) {

        try (Scanner input = new Scanner(new FileReader(fileName))) {
            while (input.hasNextLine()) {
                final ArrayList<Integer> inputFeature = new ArrayList<>();
                final String[] numbers = input.nextLine().split(",");
                for (int i = 0; i < numbers.length - 1; i++) {
                    inputFeature.add(Integer.parseInt(numbers[i]));
                }
                this.inputData.add(inputFeature);
                this.labelData.add(Integer.parseInt(numbers[numbers.length - 1]));
            }
        } catch (FileNotFoundException notFoundException) {
            notFoundException.printStackTrace();
        }
    }

    public DataReturner getData() {
        DataReturner data = new DataReturner();
        data.inputs = this.inputData;
        data.targets = this.labelData;
        return data;
    }
}