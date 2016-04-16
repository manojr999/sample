package classifier;

public class PerceptronPredictor {
	
	/**
	 * @param theta
	 * @param weights
	 * @param inputRecord
	 * @return
	 */
	public int calculateOutput(int theta, double weights[], double[] inputRecord)
	{
		
		double sum = weights[0];
		
		for (int i = 0; i < inputRecord.length; i++) {
			sum += inputRecord[i] * weights[i+1];
		}
		
		return (sum >= theta) ? 1 : 0;
	}

}
