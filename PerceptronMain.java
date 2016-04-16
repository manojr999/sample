package classifier;

public class PerceptronMain {
	
	private static final int NUMBER_OF_ATTRIBUTES = 4;

	public static void main(String[] args) {
		
		int theta = 0;
		double[] weights = new double[NUMBER_OF_ATTRIBUTES + 1];
		
		
		PerceptronTrainer perceptronTrainer = new PerceptronTrainer();
		PerceptronPredictor perceptronPredictor = new PerceptronPredictor();
		
		String trainingDataFilePath = "C:\\Users\\Manoj\\Desktop\\MS";
		String trainingDataFileName = "sample.txt";
		
		weights = perceptronTrainer.train(trainingDataFilePath, trainingDataFileName);
		
		for (int i = 0; i < weights.length; i++) {
			
			System.out.println("weight[" + i + "]: " + weights[i]);
		}
		
		double[] inputSet = new double[NUMBER_OF_ATTRIBUTES];
		inputSet[0] = 5.8;
		inputSet[1] = 4.0;
		inputSet[2] = 1.2;
		inputSet[3] = 0.2;

		System.out.println("Output Label for the given Input Data (1) :: " + perceptronPredictor.calculateOutput(theta, weights, inputSet));

	}
	
}
